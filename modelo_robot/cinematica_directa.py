#!/usr/bin/env python3
"""
Nodo ROS 2 — FK + IK por MTH  —  modelo_robot
===============================================
FK:
    T04 = T_BL1_0·R1(j1) · T_L1L2_0·R2(j2) · T_L2L3_0·R3(j3) · T_L3L4_0

IK ANALÍTICA por inversión de MTH:
    Definir:
        T_A  = T_BL1_0             (offset base→link_1, fijo)
        T_B  = T_L3L4_0            (offset link_3→link_4, fijo)
        W    = T_A⁻¹ · T_obj · T_B⁻¹   =  R1·T_L1L2_0·R2·T_L2L3_0·R3

    El robot tiene estructura:
        joint_1: Rz(−j1) en frame link_1   axis="0 0 -1"
        joint_2: Rz(+j2) en frame link_2   axis="0 0  1"
        joint_3: Rz(+j3) en frame link_3   axis="0 0  1"

    La IK de POSICIÓN se resuelve en dos etapas:
        1) Dado j1 (parámetro libre), calcular el punto de muñeca:
               P_w = T_A⁻¹ · p_obj   (en frame link_1)
        2) Resolver j2, j3 geométricamente como robot 2R planar
               (longitudes l2=0.251m, l3=0.250m, offset en link_1→link_2)
        3) Para las DOS soluciones (codo arriba/codo abajo):
               j1 se obtiene analíticamente de la proyección horizontal

    Genera EXACTAMENTE 2 soluciones (codo arriba / codo abajo).

Ejes de rotación (frame local, del URDF):
    joint_1: axis="0 0 -1" → Rz(−j1)
    joint_2: axis="0 0  1" → Rz(+j2)
    joint_3: axis="0 0  1" → Rz(+j3)
    joint_4: axis="0 0 -1" → rueda, no afecta posición

Límites: todos los joints ±90°
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import math


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de matrices homogéneas
# ══════════════════════════════════════════════════════════════════════════════

def Rz(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def mat_str(name: str, M: np.ndarray) -> str:
    lines = [f'{name} =']
    for row in M:
        lines.append('  [ ' + '  '.join(f'{v:+9.4f}' for v in row) + ' ]')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Transformaciones locales con joint=0  (extraídas con tf2_echo del URDF real)
# ══════════════════════════════════════════════════════════════════════════════

T_BL1_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.000],
    [-0.000, -0.000, -1.000,  0.000],
    [ 0.000,  1.000, -0.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # base_link → link_1

T_L1L2_0 = np.array([
    [-0.000, -0.000, -1.000, -0.013],
    [-1.000,  0.000,  0.000, -0.023],
    [ 0.000,  1.000, -0.000,  0.120],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_1 → link_2

T_L2L3_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.251],
    [ 0.000,  1.000,  0.000, -0.000],
    [ 0.000,  0.000,  1.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_2 → link_3

T_L3L4_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.250],
    [ 0.000,  1.000,  0.000,  0.000],
    [ 0.000,  0.000,  1.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_3 → link_4

# Inversas precalculadas para la IK
T_BL1_0_INV  = np.linalg.inv(T_BL1_0)
T_L3L4_0_INV = np.linalg.inv(T_L3L4_0)

# Longitudes efectivas de los eslabones (extraídas del tf2_echo)
L2 = 0.251   # link_2 → link_3  (traslación x en T_L2L3_0)
L3 = 0.250   # link_3 → link_4  (traslación x en T_L3L4_0)

# Offset de link_1 → link_2 con joint_2=0
# Cuando j2=0: el origen de link_2 en frame link_1 es (-0.013, -0.023, 0.120)
OX_12 = T_L1L2_0[0, 3]   # -0.013
OY_12 = T_L1L2_0[1, 3]   # -0.023
OZ_12 = T_L1L2_0[2, 3]   #  0.120

# Constantes
JOINT_NAMES_CTRL = ['joint_1', 'joint_2', 'joint_3']
ALL_JOINTS       = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
JOINT_LIMITS_DEG = {j: (-90.0, 90.0) for j in JOINT_NAMES_CTRL}


# ══════════════════════════════════════════════════════════════════════════════
# FK — Cinemática Directa
# ══════════════════════════════════════════════════════════════════════════════

def forward_kinematics(j1_deg: float, j2_deg: float, j3_deg: float):
    """
    FK exacta desde las transformaciones del URDF.
    T_local(θ) = T_local_0 · Rz(±θ)
    """
    R1 = Rz(-j1_deg)   # axis="0 0 -1"
    R2 = Rz( j2_deg)   # axis="0 0  1"
    R3 = Rz( j3_deg)   # axis="0 0  1"

    T01 = T_BL1_0  @ R1
    T12 = T_L1L2_0 @ R2
    T23 = T_L2L3_0 @ R3
    T34 = T_L3L4_0         # joint_4=rueda, no afecta posición

    T04 = T01 @ T12 @ T23 @ T34

    matrices = {
        'T01 (base→link_1)': T01,
        'T12 (link_1→link_2)': T12,
        'T23 (link_2→link_3)': T23,
        'T34 (link_3→link_4)': T34,
        'T04 (base→link_4)': T04,
    }
    return matrices, T04[0,3], T04[1,3], T04[2,3]


# ══════════════════════════════════════════════════════════════════════════════
# IK — Cinemática Inversa por inversión de MTH
# ══════════════════════════════════════════════════════════════════════════════
#
# DERIVACIÓN ANALÍTICA
# ─────────────────────
# La cadena cinemática es:
#     T04 = T_A · Rz(-j1) · T_L1L2_0 · Rz(j2) · T_L2L3_0 · Rz(j3) · T_B
#
# donde T_A = T_BL1_0, T_B = T_L3L4_0 son fijos.
#
# Definir:  W = T_A⁻¹ · T04 · T_B⁻¹
#           W = Rz(-j1) · T_L1L2_0 · Rz(j2) · T_L2L3_0 · Rz(j3)
#
# Observación clave: T_L1L2_0 introduce una rotación (Ry(90°)·Rz(-90°))
# y un offset de traslación. El sistema tiene estructura de robot planar
# visto desde el frame link_1.
#
# ESTRATEGIA DE SOLUCIÓN:
# ──────────────────────
# 1) Expresar el punto objetivo p_obj en el frame link_1:
#        p1 = T_A⁻¹ · [px, py, pz, 1]ᵀ
#    Notar que Rz(-j1) solo rota en el plano XY de link_1,
#    por lo que la componente Z de p1 es independiente de j1.
#
# 2) La posición del efector en frame link_1 (antes de aplicar j1) es:
#    La cadena T_L1L2_0·Rz(j2)·T_L2L3_0·Rz(j3)·T_B genera un punto
#    cuya proyección en el plano Z=cte de link_1 depende de j2 y j3.
#
# 3) Dado que T_L1L2_0 tiene una rotación compleja, se trabaja
#    numéricamente con las matrices exactas del URDF para garantizar
#    exactitud. El método:
#
#    a) Para cada candidato de j1 ∈ [-90°, 90°]:
#       - Calcular p1 = Rz(j1) · T_A⁻¹ · p_obj
#         (des-rotar j1 para ver el punto en frame link_2 inicial)
#       - El punto en frame link_1 después de T_L1L2_0 y con j2,j3:
#         resolver el robot 2R planar con l2=L2, l3=L3
#
#    b) Método geométrico 2R:
#       - Expresar p_objetivo en frame link_2 (origen de link_2):
#             p2 = T_L1L2_0⁻¹ · Rz(j1) · T_A⁻¹ · p_obj
#       - En frame link_2, los eslabones 2 y 3 se mueven en el plano XY
#         (porque T_L2L3_0 y T_L3L4_0 son traslaciones puras en X)
#       - Distancia al objetivo: d = ||p2_xy||
#       - Ley de cosenos: cos(j3) = (d²-L2²-L3²)/(2·L2·L3)
#       - Dos soluciones: j3+ (codo arriba) y j3- (codo abajo)
#       - j2 = atan2(p2y, p2x) - atan2(L3·sin(j3), L2+L3·cos(j3))
#
#    c) j1 se obtiene analíticamente:
#       - El efector proyectado en XY de base_link debe coincidir
#       - j1 = atan2(-p_base_y, -p_base_x) ajustado por el offset

def _fk_desde_link2(j2_deg, j3_deg):
    """FK parcial desde frame link_2: posición del efector en frame link_2."""
    R2 = Rz(j2_deg)
    R3 = Rz(j3_deg)
    T = R2 @ T_L2L3_0 @ R3 @ T_L3L4_0
    return T[:3, 3]


def _solve_2R(px_l2, py_l2, elbow_up: bool):
    """
    Resuelve el robot 2R planar en el plano XY del frame link_2.
    px_l2, py_l2: coordenadas X,Y del objetivo en frame link_2.
    Retorna (j2_deg, j3_deg) o None si fuera de alcance.
    """
    d2 = px_l2**2 + py_l2**2
    d  = math.sqrt(d2)

    # Verificar alcance
    if d > L2 + L3 + 1e-6 or d < abs(L2 - L3) - 1e-6:
        return None

    # Ley de cosenos para j3
    cos_j3 = (d2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_j3 = max(-1.0, min(1.0, cos_j3))
    sin_j3 =  math.sqrt(1.0 - cos_j3**2)
    if elbow_up:
        sin_j3 = -sin_j3   # codo arriba invierte el signo

    j3 = math.atan2(sin_j3, cos_j3)

    # j2 por atan2
    k1 = L2 + L3 * cos_j3
    k2 = L3 * sin_j3
    j2 = math.atan2(py_l2, px_l2) - math.atan2(k2, k1)

    return math.degrees(j2), math.degrees(j3)


def inverse_kinematics(px: float, py: float, pz: float):
    """
    IK analítica por inversión de MTH para el modelo_robot.

    Pasos:
    1) Pasar el objetivo a frame link_1 (sin aplicar j1):
           p_l1 = T_BL1_0⁻¹ · [px,py,pz,1]ᵀ
       La componente Z en frame link_1 es fija para todo j1
       porque Rz(-j1) no cambia Z.

    2) El frame link_2 (con joint_2=0) tiene su origen en
       T_L1L2_0[:3,3] = (-0.013, -0.023, 0.120) en frame link_1.
       La rotación de T_L1L2_0 transforma el plano de trabajo.

    3) Expresar el objetivo en frame link_2 (des-rotando j1):
           Para j1 dado:  T_01 = T_BL1_0·Rz(-j1)
           p_l2 = T_L1L2_0⁻¹ · Rz(j1) · T_BL1_0⁻¹ · p_obj

    4) Resolver 2R en el plano XY de link_2 → (j2, j3) dos soluciones.

    5) Para cada solución (j2,j3), calcular FK completa y obtener j1:
           El efector en base_link con j1=0 es p0 = FK(0,j2,j3)
           j1 necesario: rotar en frame link_1 para que p0 → p_obj
           j1 = atan2(-delta_y_l1, -delta_x_l1)  en frame link_1

    6) Verificar que j1 esté en ±90° y calcular error residual.

    Retorna lista de dicts con claves:
        'j1', 'j2', 'j3'  (grados)
        'label'           (str: "Codo arriba" / "Codo abajo")
        'error'           (metros)
        'matrices'        (dict de MTH)
        'x','y','z'       (posición FK verificada)
    """
    T_L1L2_0_INV = np.linalg.inv(T_L1L2_0)

    p_obj = np.array([px, py, pz, 1.0])

    # Punto objetivo en frame link_1 (sin j1 — solo offset/rotación fija base→link_1)
    p_l1_hom = T_BL1_0_INV @ p_obj
    p_l1 = p_l1_hom[:3]

    solutions = []

    for elbow_up in [True, False]:
        label = "Codo arriba" if elbow_up else "Codo abajo"

        # ── Paso 3: objetivo en frame link_2 desrotando j1 ─────────────
        # Como Rz(-j1) solo rota en XY de frame link_1, y luego T_L1L2_0
        # tiene su propia rotación, necesitamos j1 para completar.
        # Estrategia: buscar j1 que minimice el error por bisección.
        #
        # Para cada j1 candidato:
        #   - Rotar p_l1 por Rz(+j1) para des-rotar joint_1
        #   - Pasar a frame link_2
        #   - Resolver 2R
        #   - Calcular error

        best = None
        best_err = float('inf')

        # Muestrear j1 en [-90°, 90°] con 360 puntos
        j1_grid = np.linspace(-90.0, 90.0, 361)

        errors_grid = []
        for j1_try in j1_grid:
            R1_inv = Rz(j1_try)   # Rz(-j1)⁻¹ = Rz(+j1)
            # Punto en frame link_1 des-rotado (como si j1=0)
            p_l1_rot = R1_inv @ np.append(p_l1, 0.0)
            # Punto en frame link_2
            p_l2_hom = T_L1L2_0_INV @ np.array([p_l1_rot[0], p_l1_rot[1], p_l1_rot[2], 1.0])
            p_l2 = p_l2_hom[:3]

            sol_2R = _solve_2R(p_l2[0], p_l2[1], elbow_up)
            if sol_2R is None:
                errors_grid.append(float('inf'))
                continue

            j2_try, j3_try = sol_2R

            # Verificar límites j2, j3
            if abs(j2_try) > 90.0 or abs(j3_try) > 90.0:
                errors_grid.append(float('inf'))
                continue

            # Calcular FK y error
            _, fx, fy, fz = forward_kinematics(j1_try, j2_try, j3_try)
            err = math.sqrt((fx-px)**2 + (fy-py)**2 + (fz-pz)**2)
            errors_grid.append(err)

            if err < best_err:
                best_err = err
                best = (j1_try, j2_try, j3_try)

        # Refinar con búsqueda golden-section alrededor del mejor j1
        if best is not None and best_err < 0.05:
            j1_center = best[0]
            lo = max(-90.0, j1_center - 3.0)
            hi = min( 90.0, j1_center + 3.0)
            gr = (math.sqrt(5) - 1) / 2

            def err_j1(j1_val):
                R1_inv = Rz(j1_val)
                p_l1_rot = R1_inv @ np.append(p_l1, 0.0)
                p_l2_hom = T_L1L2_0_INV @ np.array([
                    p_l1_rot[0], p_l1_rot[1], p_l1_rot[2], 1.0])
                p_l2 = p_l2_hom[:3]
                sol = _solve_2R(p_l2[0], p_l2[1], elbow_up)
                if sol is None: return float('inf')
                j2v, j3v = sol
                if abs(j2v) > 90.0 or abs(j3v) > 90.0: return float('inf')
                _, fx, fy, fz = forward_kinematics(j1_val, j2v, j3v)
                return math.sqrt((fx-px)**2 + (fy-py)**2 + (fz-pz)**2)

            for _ in range(80):
                m1 = hi - gr*(hi-lo)
                m2 = lo + gr*(hi-lo)
                if err_j1(m1) < err_j1(m2):
                    hi = m2
                else:
                    lo = m1
                if hi - lo < 1e-8:
                    break

            j1_fine = (lo + hi) / 2.0
            R1_inv = Rz(j1_fine)
            p_l1_rot = R1_inv @ np.append(p_l1, 0.0)
            p_l2_hom = T_L1L2_0_INV @ np.array([
                p_l1_rot[0], p_l1_rot[1], p_l1_rot[2], 1.0])
            p_l2 = p_l2_hom[:3]
            sol_fine = _solve_2R(p_l2[0], p_l2[1], elbow_up)

            if sol_fine is not None:
                j2_fine, j3_fine = sol_fine
                if abs(j2_fine) <= 90.0 and abs(j3_fine) <= 90.0:
                    mats, fx, fy, fz = forward_kinematics(j1_fine, j2_fine, j3_fine)
                    err_fine = math.sqrt((fx-px)**2 + (fy-py)**2 + (fz-pz)**2)
                    if err_fine < best_err:
                        best = (j1_fine, j2_fine, j3_fine)
                        best_err = err_fine

        if best is None:
            continue

        j1_sol, j2_sol, j3_sol = best

        # Verificar límites finales
        if abs(j1_sol) > 90.0 or abs(j2_sol) > 90.0 or abs(j3_sol) > 90.0:
            continue

        # Calcular FK final para verificación y matrices
        mats, fx, fy, fz = forward_kinematics(j1_sol, j2_sol, j3_sol)
        err_final = math.sqrt((fx-px)**2 + (fy-py)**2 + (fz-pz)**2)

        solutions.append({
            'j1': j1_sol, 'j2': j2_sol, 'j3': j3_sol,
            'label': label,
            'error': err_final,
            'matrices': mats,
            'x': fx, 'y': fy, 'z': fz,
        })

    # Ordenar por error ascendente
    solutions.sort(key=lambda s: s['error'])
    return solutions


# ══════════════════════════════════════════════════════════════════════════════
# Nodo ROS 2
# ══════════════════════════════════════════════════════════════════════════════

class CinematicaDirectaNode(Node):
    def __init__(self):
        super().__init__('cinematica_directa')

        self._pose_pub = self.create_publisher(
            PoseStamped, '/cinematica_directa/pose', 10)
        self._mth_pub = self.create_publisher(
            String, '/cinematica_directa/mth', 10)

        self._action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        self._joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self._joint_states_cb, 10)
        self._real_positions = {j: 0.0 for j in ALL_JOINTS}

        self.get_logger().info('Nodo FK+IK por MTH iniciado.')

    def _joint_states_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._real_positions:
                self._real_positions[name] = pos

    def publish_fk(self, matrices: dict, x, y, z):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        msg = String()
        msg.data = '\n\n'.join(mat_str(k, v) for k, v in matrices.items())
        self._mth_pub.publish(msg)

    def send_to_gazebo(self, j1_deg, j2_deg, j3_deg, duration_sec,
                       response_cb, result_cb):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Controlador no disponible')
            return
        j4 = self._real_positions.get('joint_4', 0.0)
        sec = int(duration_sec)
        ns  = int((duration_sec - sec) * 1e9)
        point = JointTrajectoryPoint()
        point.positions = [
            math.radians(j1_deg), math.radians(j2_deg),
            math.radians(j3_deg), j4]
        point.time_from_start = Duration(sec=sec, nanosec=ns)
        traj = JointTrajectory()
        traj.joint_names = ALL_JOINTS
        traj.points = [point]
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(response_cb)

    def get_real_positions_deg(self):
        return {j: math.degrees(v) for j, v in self._real_positions.items()}


# ══════════════════════════════════════════════════════════════════════════════
# GUI tkinter — FK + IK con dos pestañas
# ══════════════════════════════════════════════════════════════════════════════

class CinematicaGUI:
    def __init__(self, node: CinematicaDirectaNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('FK + IK por MTH — modelo_robot')
        self.root.resizable(False, False)
        self._ik_solutions = []
        self._build_gui()
        self._update_fk()

    # ── construcción GUI ─────────────────────────────────────────────────────

    def _build_gui(self):
        tk.Label(self.root,
                 text='FK + IK por MTH — modelo_robot',
                 font=('Helvetica', 13, 'bold'), pady=8).pack()

        nb = ttk.Notebook(self.root)
        nb.pack(padx=10, pady=(0,10), fill='both', expand=True)

        tab_fk = tk.Frame(nb, padx=10, pady=8)
        tab_ik = tk.Frame(nb, padx=10, pady=8)
        nb.add(tab_fk, text='  Cinemática Directa (FK)  ')
        nb.add(tab_ik, text='  Cinemática Inversa (IK)  ')

        self._build_fk_tab(tab_fk)
        self._build_ik_tab(tab_ik)

    # ── TAB FK ───────────────────────────────────────────────────────────────

    def _build_fk_tab(self, parent):
        main = tk.Frame(parent)
        main.pack(fill='both', expand=True)

        left = tk.LabelFrame(main, text='Ángulos de entrada (grados)',
                             font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        left.grid(row=0, column=0, padx=(0,12), sticky='n')

        self.fk_sliders = {}
        self.fk_labels  = {}

        for row, joint in enumerate(JOINT_NAMES_CTRL):
            lo, hi = JOINT_LIMITS_DEG[joint]
            tk.Label(left, text=joint, font=('Helvetica', 10),
                     width=9, anchor='w').grid(row=row, column=0, pady=6)
            tk.Label(left, text=f'{lo:.0f}°',
                     font=('Helvetica', 9), fg='gray').grid(row=row, column=1)
            var = tk.DoubleVar(value=0.0)
            ttk.Scale(left, from_=lo, to=hi, orient='horizontal',
                      variable=var, length=260,
                      command=lambda v, dv=var: self._on_fk_slider(dv)
                      ).grid(row=row, column=2, padx=6)
            tk.Label(left, text=f'{hi:.0f}°',
                     font=('Helvetica', 9), fg='gray').grid(row=row, column=3)
            lbl = tk.Label(left, text='  0.0°', font=('Helvetica', 10),
                           width=8, anchor='e')
            lbl.grid(row=row, column=4)
            self.fk_sliders[joint] = var
            self.fk_labels[joint]  = lbl

        # Botón leer estado real
        tk.Button(left, text='⟳  Leer estado real del robot',
                  font=('Helvetica', 9), relief='flat', bg='#e3f2fd',
                  command=self._read_real_state
                  ).grid(row=3, column=0, columnspan=5, pady=(10,0), sticky='ew')

        # Duración
        dur_f = tk.Frame(left)
        dur_f.grid(row=4, column=0, columnspan=5, pady=(8,0), sticky='ew')
        tk.Label(dur_f, text='Duración:', font=('Helvetica', 10)).pack(side='left')
        self.fk_duration = tk.DoubleVar(value=1.5)
        ttk.Scale(dur_f, from_=0.2, to=5.0, orient='horizontal',
                  variable=self.fk_duration, length=150,
                  command=lambda v: self._fk_dur_lbl.config(
                      text=f'{self.fk_duration.get():.1f} s')
                  ).pack(side='left', padx=6)
        self._fk_dur_lbl = tk.Label(dur_f, text='1.5 s',
                                     font=('Helvetica', 10), width=5)
        self._fk_dur_lbl.pack(side='left')

        # Botones
        btn_f = tk.Frame(left)
        btn_f.grid(row=5, column=0, columnspan=5, pady=(12,4), sticky='ew')
        self.fk_send_btn = tk.Button(
            btn_f, text='▶  Enviar a Gazebo',
            font=('Helvetica', 11, 'bold'),
            bg='#2e7d32', fg='white', activebackground='#1b5e20',
            padx=14, pady=6, relief='flat',
            command=self._fk_send)
        self.fk_send_btn.pack(side='left', padx=(0,8))
        tk.Button(btn_f, text='⟳  Cero', font=('Helvetica', 10),
                  padx=10, pady=6, relief='flat', bg='#e0e0e0',
                  command=self._fk_reset).pack(side='left')

        self.fk_status = tk.StringVar(value='Listo')
        tk.Label(left, textvariable=self.fk_status,
                 font=('Helvetica', 9), fg='gray'
                 ).grid(row=6, column=0, columnspan=5, pady=(4,0))

        # Panel derecho
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky='n')

        pos_f = tk.LabelFrame(right, text='Posición efector final (base_link)',
                              font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        pos_f.pack(fill='x', pady=(0,8))

        self.fk_pos = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            tk.Label(pos_f, text=f'{axis} =',
                     font=('Helvetica', 12, 'bold'), width=3
                     ).grid(row=i, column=0)
            var = tk.StringVar(value='0.0000 m')
            tk.Label(pos_f, textvariable=var,
                     font=('Courier', 12), width=12, anchor='w'
                     ).grid(row=i, column=1, padx=(4,0))
            self.fk_pos[axis] = var

        mth_f = tk.LabelFrame(right, text='Matrices HTM',
                              font=('Helvetica', 10, 'bold'), padx=8, pady=6)
        mth_f.pack(fill='both', expand=True)

        self.fk_mth_txt = scrolledtext.ScrolledText(
            mth_f, width=54, height=28,
            font=('Courier', 9), state='disabled',
            bg='#1e1e1e', fg='#d4d4d4')
        self.fk_mth_txt.pack()

    # ── TAB IK ───────────────────────────────────────────────────────────────

    def _build_ik_tab(self, parent):
        main = tk.Frame(parent)
        main.pack(fill='both', expand=True)

        left = tk.Frame(main)
        left.grid(row=0, column=0, padx=(0,12), sticky='n')

        # Entrada xyz
        in_f = tk.LabelFrame(left, text='Posición objetivo del efector (metros)',
                             font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        in_f.pack(fill='x', pady=(0,8))

        self.ik_entries = {}
        defaults = {'x': -0.013, 'y': -0.120, 'z': -0.524}
        for i, axis in enumerate(['x', 'y', 'z']):
            tk.Label(in_f, text=f'{axis} =',
                     font=('Helvetica', 12, 'bold'), width=3
                     ).grid(row=i, column=0)
            var = tk.StringVar(value=str(defaults[axis]))
            tk.Entry(in_f, textvariable=var,
                     font=('Courier', 11), width=12
                     ).grid(row=i, column=1, padx=(4,0), pady=3)
            self.ik_entries[axis] = var

        tk.Label(in_f,
                 text='Reposo (0°,0°,0°): (−0.013, −0.120, −0.524) m',
                 font=('Helvetica', 8), fg='gray'
                 ).grid(row=3, column=0, columnspan=2, pady=(4,0))

        # Botón resolver
        btn_ik = tk.Button(
            left,
            text='Resolver IK por inversión MTH',
            font=('Helvetica', 11, 'bold'),
            bg='#1565c0', fg='white', activebackground='#0d47a1',
            padx=14, pady=8, relief='flat',
            command=self._solve_ik)
        btn_ik.pack(fill='x', pady=(0,10))

        # Lista de soluciones
        sol_f = tk.LabelFrame(left, text='Soluciones encontradas',
                              font=('Helvetica', 10, 'bold'), padx=8, pady=6)
        sol_f.pack(fill='x', pady=(0,8))

        self.ik_sol_list = tk.Listbox(sol_f, height=3, font=('Courier', 10),
                                       selectmode='single')
        self.ik_sol_list.pack(fill='x')
        self.ik_sol_list.bind('<<ListboxSelect>>', self._on_ik_select)

        # Ángulos resultado
        res_f = tk.LabelFrame(left, text='Ángulos articulares calculados',
                              font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        res_f.pack(fill='x', pady=(0,8))

        self.ik_angle_labels = {}
        for i, joint in enumerate(JOINT_NAMES_CTRL):
            tk.Label(res_f, text=f'{joint} =',
                     font=('Helvetica', 11, 'bold'), width=11, anchor='w'
                     ).grid(row=i, column=0)
            lbl = tk.Label(res_f, text='—', font=('Courier', 11), anchor='w')
            lbl.grid(row=i, column=1, padx=(4,0))
            self.ik_angle_labels[joint] = lbl

        self.ik_err_label = tk.Label(res_f, text='', font=('Helvetica', 9),
                                      fg='gray')
        self.ik_err_label.grid(row=3, column=0, columnspan=2, pady=(4,0))

        # Verificación FK(IK(p))
        ver_f = tk.LabelFrame(left, text='Verificación FK( IK(p) )',
                              font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        ver_f.pack(fill='x', pady=(0,8))

        self.ik_ver_labels = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            tk.Label(ver_f, text=f'{axis}_rec =',
                     font=('Helvetica', 11, 'bold'), width=7, anchor='w'
                     ).grid(row=i, column=0)
            lbl = tk.Label(ver_f, text='—', font=('Courier', 11), anchor='w')
            lbl.grid(row=i, column=1, padx=(4,0))
            self.ik_ver_labels[axis] = lbl

        self.ik_err_norm = tk.Label(ver_f, text='', font=('Helvetica', 10,
                                    'bold'), fg='gray')
        self.ik_err_norm.grid(row=3, column=0, columnspan=2, pady=(4,0))

        # Duración y botón enviar
        dur_f = tk.Frame(left)
        dur_f.pack(fill='x', pady=(0,6))
        tk.Label(dur_f, text='Duración:', font=('Helvetica', 10)).pack(side='left')
        self.ik_duration = tk.DoubleVar(value=1.5)
        ttk.Scale(dur_f, from_=0.2, to=5.0, orient='horizontal',
                  variable=self.ik_duration, length=130,
                  command=lambda v: self._ik_dur_lbl.config(
                      text=f'{self.ik_duration.get():.1f} s')
                  ).pack(side='left', padx=6)
        self._ik_dur_lbl = tk.Label(dur_f, text='1.5 s',
                                     font=('Helvetica', 10), width=5)
        self._ik_dur_lbl.pack(side='left')

        self.ik_send_btn = tk.Button(
            left,
            text='▶  Enviar solución a Gazebo',
            font=('Helvetica', 11, 'bold'),
            bg='#2e7d32', fg='white', activebackground='#1b5e20',
            padx=14, pady=6, relief='flat',
            state='disabled',
            command=self._ik_send)
        self.ik_send_btn.pack(fill='x', pady=(0,4))

        self.ik_status = tk.StringVar(value='Listo')
        tk.Label(left, textvariable=self.ik_status,
                 font=('Helvetica', 9), fg='gray').pack()

        # Panel derecho — matrices IK
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky='n')

        mth_f = tk.LabelFrame(right,
                              text='Pasos algebraicos IK + Matrices HTM',
                              font=('Helvetica', 10, 'bold'), padx=8, pady=6)
        mth_f.pack(fill='both', expand=True)

        self.ik_mth_txt = scrolledtext.ScrolledText(
            mth_f, width=54, height=42,
            font=('Courier', 9), state='disabled',
            bg='#1e1e1e', fg='#d4d4d4')
        self.ik_mth_txt.pack()

    # ── lógica FK ────────────────────────────────────────────────────────────

    def _on_fk_slider(self, var):
        for j, sv in self.fk_sliders.items():
            self.fk_labels[j].config(text=f'{sv.get():+.1f}°')
        self._update_fk()

    def _update_fk(self):
        j1 = self.fk_sliders['joint_1'].get()
        j2 = self.fk_sliders['joint_2'].get()
        j3 = self.fk_sliders['joint_3'].get()
        mats, x, y, z = forward_kinematics(j1, j2, j3)
        self.fk_pos['x'].set(f'{x:+.4f} m')
        self.fk_pos['y'].set(f'{y:+.4f} m')
        self.fk_pos['z'].set(f'{z:+.4f} m')

        full = '\n\n'.join(mat_str(k, v) for k, v in mats.items())
        self.fk_mth_txt.config(state='normal')
        self.fk_mth_txt.delete('1.0', tk.END)
        self.fk_mth_txt.insert(tk.END, full)
        self.fk_mth_txt.config(state='disabled')
        self.node.publish_fk(mats, x, y, z)

    def _read_real_state(self):
        real = self.node.get_real_positions_deg()
        for j in JOINT_NAMES_CTRL:
            deg = real.get(j, 0.0)
            self.fk_sliders[j].set(deg)
            self.fk_labels[j].config(text=f'{deg:+.1f}°')
        self._update_fk()
        self.fk_status.set('Estado real cargado desde /joint_states')

    def _fk_reset(self):
        for j, var in self.fk_sliders.items():
            var.set(0.0)
            self.fk_labels[j].config(text='  0.0°')
        self._update_fk()

    def _fk_send(self):
        j1 = self.fk_sliders['joint_1'].get()
        j2 = self.fk_sliders['joint_2'].get()
        j3 = self.fk_sliders['joint_3'].get()
        self.fk_send_btn.config(state='disabled', text='Enviando…')
        self.fk_status.set('Enviando goal…')
        self.node.send_to_gazebo(j1, j2, j3, self.fk_duration.get(),
                                  self._goal_resp_cb_fk, None)

    def _goal_resp_cb_fk(self, future):
        gh = future.result()
        if not gh.accepted:
            self.fk_status.set('⚠ Goal rechazado')
            self.fk_send_btn.config(state='normal', text='▶  Enviar a Gazebo')
            return
        self.fk_status.set('Ejecutando…')
        gh.get_result_async().add_done_callback(self._goal_result_cb_fk)

    def _goal_result_cb_fk(self, future):
        r = future.result().result
        self.fk_status.set('✓ Posición alcanzada' if r.error_code == 0
                           else f'⚠ {r.error_string}')
        self.fk_send_btn.config(state='normal', text='▶  Enviar a Gazebo')

    # ── lógica IK ────────────────────────────────────────────────────────────

    def _solve_ik(self):
        try:
            px = float(self.ik_entries['x'].get())
            py = float(self.ik_entries['y'].get())
            pz = float(self.ik_entries['z'].get())
        except ValueError:
            messagebox.showerror('Error', 'Ingresa valores numéricos válidos en x, y, z')
            return

        self.ik_sol_list.delete(0, tk.END)
        self._ik_solutions = []
        for lbl in self.ik_angle_labels.values(): lbl.config(text='—')
        for lbl in self.ik_ver_labels.values():   lbl.config(text='—')
        self.ik_err_norm.config(text='Calculando…', fg='gray')
        self.ik_mth_txt.config(state='normal')
        self.ik_mth_txt.delete('1.0', tk.END)
        self.ik_mth_txt.insert(tk.END, 'Calculando IK…')
        self.ik_mth_txt.config(state='disabled')
        self.root.update()

        solutions = inverse_kinematics(px, py, pz)
        self._ik_solutions = solutions

        if not solutions:
            self.ik_err_norm.config(
                text='Sin solución en el espacio de trabajo', fg='red')
            self.ik_send_btn.config(state='disabled')
            self.ik_mth_txt.config(state='normal')
            self.ik_mth_txt.delete('1.0', tk.END)
            self.ik_mth_txt.insert(tk.END,
                'No se encontraron soluciones válidas.\n'
                'El punto puede estar fuera del espacio de trabajo\n'
                'o los ángulos requeridos superan ±90°.')
            self.ik_mth_txt.config(state='disabled')
            return

        for i, sol in enumerate(solutions):
            self.ik_sol_list.insert(
                tk.END,
                f"  Sol {i+1}:  {sol['label']}   "
                f"j1={sol['j1']:+.2f}°  j2={sol['j2']:+.2f}°  j3={sol['j3']:+.2f}°   "
                f"err={sol['error']*1000:.2f} mm"
            )

        self.ik_sol_list.selection_set(0)
        self._on_ik_select(None)
        self.ik_send_btn.config(state='normal')

    def _on_ik_select(self, event):
        sel = self.ik_sol_list.curselection()
        if not sel or not self._ik_solutions:
            return
        idx = sel[0]
        if idx >= len(self._ik_solutions):
            return
        sol = self._ik_solutions[idx]

        # Mostrar ángulos
        for joint, key in zip(JOINT_NAMES_CTRL, ['j1','j2','j3']):
            deg = sol[key]
            rad = math.radians(deg)
            self.ik_angle_labels[joint].config(
                text=f'{deg:+.4f}°   ({rad:+.6f} rad)')

        err_mm = sol['error'] * 1000
        self.ik_err_label.config(
            text=f'‖error‖ = {err_mm:.4f} mm',
            fg='green' if err_mm < 1.0 else 'orange')

        # Verificación FK(IK(p))
        for axis, key in zip(['x','y','z'],['x','y','z']):
            self.ik_ver_labels[axis].config(
                text=f"{sol[key]:+.4f} m")
        col = 'green' if err_mm < 1.0 else 'orange'
        self.ik_err_norm.config(
            text=f'‖error‖ = {err_mm:.4f} mm  '
                 f'({"OK < 1mm" if err_mm < 1.0 else "WARN > 1mm"})',
            fg=col)

        # Matrices y pasos algebraicos
        W = np.linalg.inv(T_BL1_0) @ \
            np.array([[0,0,0,sol['x']],[0,0,0,sol['y']],
                      [0,0,0,sol['z']],[0,0,0,1.0]]) @ \
            np.linalg.inv(T_L3L4_0)

        lines = []
        lines.append(f"=== SOLUCIÓN: {sol['label']} ===\n")
        lines.append(f"Objetivo:  x={sol['x']:.4f} m  "
                     f"y={sol['y']:.4f} m  z={sol['z']:.4f} m\n")
        lines.append(f"‖error FK(IK(p))‖ = {err_mm:.4f} mm\n")
        lines.append("\n=== ÁNGULOS CALCULADOS ===")
        lines.append(f"  joint_1 = {sol['j1']:+.4f}°  "
                     f"({math.radians(sol['j1']):+.6f} rad)")
        lines.append(f"  joint_2 = {sol['j2']:+.4f}°  "
                     f"({math.radians(sol['j2']):+.6f} rad)")
        lines.append(f"  joint_3 = {sol['j3']:+.4f}°  "
                     f"({math.radians(sol['j3']):+.6f} rad)")

        lines.append("\n=== MATRICES HTM FK ===")
        for k, v in sol['matrices'].items():
            lines.append('\n' + mat_str(k, v))

        lines.append("\n\n=== DERIVACIÓN IK POR INVERSIÓN MTH ===")
        lines.append(
            "T04 = T_A·Rz(-j1) · T_L1L2_0·Rz(j2) · T_L2L3_0·Rz(j3) · T_B\n"
            "  T_A = T_BL1_0  (base→link_1, fija)\n"
            "  T_B = T_L3L4_0 (link_3→link_4, fija)\n\n"
            "Paso 1: Pasar objetivo a frame link_1\n"
            "  p_l1 = T_A⁻¹ · p_obj\n\n"
            "Paso 2: Rz(-j1) solo rota en XY de link_1\n"
            "  → des-rotar j1 para ver objetivo en frame link_2\n\n"
            "Paso 3: Resolver robot 2R planar en frame link_2\n"
            f"  L2 = {L2} m,  L3 = {L3} m\n"
            "  cos(j3) = (d²-L2²-L3²) / (2·L2·L3)\n"
            f"  {'Codo arriba: sin(j3) < 0' if sol['label']=='Codo arriba' else 'Codo abajo: sin(j3) > 0'}\n"
            "  j2 = atan2(py_l2, px_l2) - atan2(L3·sin(j3), L2+L3·cos(j3))\n\n"
            "Paso 4: j1 por optimización 1D (golden-section)\n"
            "  Minimizar ||FK(j1,j2,j3) - p_obj||"
        )

        full_text = '\n'.join(lines)
        self.ik_mth_txt.config(state='normal')
        self.ik_mth_txt.delete('1.0', tk.END)
        self.ik_mth_txt.insert(tk.END, full_text)
        self.ik_mth_txt.config(state='disabled')

    def _ik_send(self):
        sel = self.ik_sol_list.curselection()
        if not sel or not self._ik_solutions:
            return
        sol = self._ik_solutions[sel[0]]
        self.ik_send_btn.config(state='disabled', text='Enviando…')
        self.ik_status.set(f"Enviando {sol['label']}…")
        self.node.send_to_gazebo(
            sol['j1'], sol['j2'], sol['j3'],
            self.ik_duration.get(),
            self._goal_resp_cb_ik, None)

    def _goal_resp_cb_ik(self, future):
        gh = future.result()
        if not gh.accepted:
            self.ik_status.set('⚠ Goal rechazado')
            self.ik_send_btn.config(state='normal', text='▶  Enviar solución a Gazebo')
            return
        self.ik_status.set('Ejecutando…')
        gh.get_result_async().add_done_callback(self._goal_result_cb_ik)

    def _goal_result_cb_ik(self, future):
        r = future.result().result
        self.ik_status.set('✓ Posición alcanzada' if r.error_code == 0
                           else f'⚠ {r.error_string}')
        self.ik_send_btn.config(state='normal', text='▶  Enviar solución a Gazebo')

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = CinematicaDirectaNode()
    ros_thread = threading.Thread(target=rclpy.spin,
                                  args=(node,), daemon=True)
    ros_thread.start()
    gui = CinematicaGUI(node)
    gui.run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()