#!/usr/bin/env python3
"""
Nodo ROS 2 — Cinemática Directa del modelo_robot
=================================================
FK construida directamente desde las transformaciones del URDF
verificadas con tf2_echo. No usa tabla DH aproximada.

Transformaciones locales (joints en cero, extraídas con tf2_echo):
  base_link → link_1: t=(0,0,0),            R=Rx(90°)
  link_1    → link_2: t=(-0.013,-0.023,0.120), R=Ry(90°)·Rz(-90°)
  link_2    → link_3: t=(0.251,0,0),         R=I
  link_3    → link_4: t=(0.250,0,0),         R=I

Ejes de rotación de cada joint (frame local, según URDF):
  joint_1 (base→link_1): eje Z local de link_1  → Rz(-θ₁)  [axis xyz="0 0 -1"]
  joint_2 (link_1→link_2): eje Z local de link_2 → Rz(θ₂)  [axis xyz="0 0 1"]
  joint_3 (link_2→link_3): eje Z local de link_3 → Rz(θ₃)  [axis xyz="0 0 1"]
  joint_4 (link_3→link_4): eje Z local           → Rz(-θ₄) [axis xyz="0 0 -1"] (rueda)

Publica:
  /cinematica_directa/pose  → geometry_msgs/PoseStamped
  /cinematica_directa/mth   → std_msgs/String
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
from tkinter import ttk, scrolledtext
import math


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de matrices homogéneas
# ══════════════════════════════════════════════════════════════════════════════

def Rx(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=float)

def Ry(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=float)

def Rz(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=float)

def Trans(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[0,3], T[1,3], T[2,3] = x, y, z
    return T

def mat_str(name: str, M: np.ndarray) -> str:
    lines = [f'{name} =']
    for row in M:
        lines.append('  [ ' + '  '.join(f'{v:9.4f}' for v in row) + ' ]')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Cinemática directa — basada en tf2_echo del URDF real
# ══════════════════════════════════════════════════════════════════════════════
#
# Cada T_local = T_offset · R_joint
# donde T_offset es la transformación con joint=0 (extraída con tf2_echo)
# y R_joint es la rotación del joint alrededor de su eje local.
#
# Ejes según URDF:
#   joint_1: axis="0 0 -1"  → Rz(-θ₁)
#   joint_2: axis="0 0  1"  → Rz(+θ₂)
#   joint_3: axis="0 0  1"  → Rz(+θ₃)
#   joint_4: axis="0 0 -1"  → Rz(-θ₄)  (rueda, no afecta posición)

# Transformaciones locales con joint=0 (matrices 4x4 extraídas de tf2_echo)
T_BL1_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.000],
    [-0.000, -0.000, -1.000,  0.000],
    [ 0.000,  1.000, -0.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # base_link → link_1 con joint_1=0

T_L1L2_0 = np.array([
    [-0.000, -0.000, -1.000, -0.013],
    [-1.000,  0.000,  0.000, -0.023],
    [ 0.000,  1.000, -0.000,  0.120],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_1 → link_2 con joint_2=0

T_L2L3_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.251],
    [ 0.000,  1.000,  0.000, -0.000],
    [ 0.000,  0.000,  1.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_2 → link_3 con joint_3=0

T_L3L4_0 = np.array([
    [ 1.000, -0.000,  0.000,  0.250],
    [ 0.000,  1.000,  0.000,  0.000],
    [ 0.000,  0.000,  1.000,  0.000],
    [ 0.000,  0.000,  0.000,  1.000],
], dtype=float)  # link_3 → link_4 con joint_4=0


def forward_kinematics(j1_deg: float, j2_deg: float, j3_deg: float):
    """
    Calcula la FK del efector final (link_4) respecto a base_link.

    Cada transformación local se obtiene como:
        T_local(θ) = T_local_0 · R_joint(θ)
    donde R_joint rota alrededor del eje Z local del frame hijo
    (todos los joints son revolute con eje Z en su frame local,
    con signo según el atributo axis del URDF).

    Returns:
        matrices: dict con T01..T34 y T04
        x, y, z: posición del efector final en base_link
    """
    # Rotaciones de cada joint en su frame local
    # joint_1: axis="0 0 -1" → Rz(-θ₁)
    # joint_2: axis="0 0  1" → Rz(+θ₂)
    # joint_3: axis="0 0  1" → Rz(+θ₃)
    R1 = Rz(-j1_deg)
    R2 = Rz( j2_deg)
    R3 = Rz( j3_deg)

    # Transformaciones locales con el joint aplicado
    T01 = T_BL1_0 @ R1
    T12 = T_L1L2_0 @ R2
    T23 = T_L2L3_0 @ R3
    T34 = T_L3L4_0          # joint_4 es la rueda, no afecta posición

    # Transformación total base_link → link_4
    T04 = T01 @ T12 @ T23 @ T34

    matrices = {
        'T01 (base→link_1)': T01,
        'T12 (link_1→link_2)': T12,
        'T23 (link_2→link_3)': T23,
        'T34 (link_3→link_4)': T34,
        'T04 (base→link_4)': T04,
    }

    x, y, z = T04[0,3], T04[1,3], T04[2,3]
    return matrices, x, y, z


# ══════════════════════════════════════════════════════════════════════════════
# Constantes de control
# ══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES_CTRL = ['joint_1', 'joint_2', 'joint_3']
ALL_JOINTS       = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
JOINT_LIMITS_DEG = {
    'joint_1': (-90.0, 90.0),
    'joint_2': (-90.0, 90.0),
    'joint_3': (-90.0, 90.0),
}


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

        self.get_logger().info('Nodo cinemática directa (FK desde URDF) iniciado.')

    def _joint_states_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self._real_positions:
                self._real_positions[name] = pos

    def publish_fk(self, matrices: dict, x: float, y: float, z: float):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        text_parts = [mat_str(k, v) for k, v in matrices.items()]
        msg = String()
        msg.data = '\n\n'.join(text_parts)
        self._mth_pub.publish(msg)

    def send_to_gazebo(self, j1_deg, j2_deg, j3_deg, duration_sec,
                       response_cb, result_cb):
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Controlador no disponible')
            return

        j4_current = self._real_positions.get('joint_4', 0.0)

        point = JointTrajectoryPoint()
        point.positions = [
            math.radians(j1_deg),
            math.radians(j2_deg),
            math.radians(j3_deg),
            j4_current,
        ]
        sec = int(duration_sec)
        ns  = int((duration_sec - sec) * 1e9)
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
# GUI tkinter
# ══════════════════════════════════════════════════════════════════════════════

class CinematicaGUI:
    def __init__(self, node: CinematicaDirectaNode):
        self.node = node
        self.root = tk.Tk()
        self.root.title('Cinemática Directa — modelo_robot')
        self.root.resizable(False, False)
        self._build_gui()
        self._update_fk()

    def _build_gui(self):
        tk.Label(self.root,
                 text='Cinemática Directa — modelo_robot',
                 font=('Helvetica', 13, 'bold'), pady=8).pack()

        main = tk.Frame(self.root, padx=16, pady=4)
        main.pack()

        # ── Panel izquierdo ──────────────────────────────────────
        left = tk.LabelFrame(main, text='Ángulos de entrada (grados)',
                             font=('Helvetica', 10, 'bold'), padx=12, pady=8)
        left.grid(row=0, column=0, padx=(0,12), sticky='n')

        self.sliders     = {}
        self.angle_labels = {}

        for row, joint in enumerate(JOINT_NAMES_CTRL):
            lo, hi = JOINT_LIMITS_DEG[joint]
            tk.Label(left, text=joint, font=('Helvetica', 10),
                     width=9, anchor='w').grid(row=row, column=0, pady=6)
            tk.Label(left, text=f'{lo:.0f}°',
                     font=('Helvetica', 9), fg='gray').grid(row=row, column=1)

            var = tk.DoubleVar(value=0.0)
            ttk.Scale(left, from_=lo, to=hi, orient='horizontal',
                      variable=var, length=260,
                      command=lambda v, dv=var: self._on_slider(dv)
                      ).grid(row=row, column=2, padx=6)

            tk.Label(left, text=f'{hi:.0f}°',
                     font=('Helvetica', 9), fg='gray').grid(row=row, column=3)

            lbl = tk.Label(left, text='  0.0°', font=('Helvetica', 10),
                           width=8, anchor='e')
            lbl.grid(row=row, column=4)

            self.sliders[joint]       = var
            self.angle_labels[joint]  = lbl

        # Botón leer estado real
        tk.Button(left, text='⟳  Leer estado real del robot',
                  font=('Helvetica', 9), relief='flat', bg='#e3f2fd',
                  command=self._read_real_state
                  ).grid(row=len(JOINT_NAMES_CTRL), column=0,
                         columnspan=5, pady=(10,0), sticky='ew')

        # Duración
        dur_f = tk.Frame(left)
        dur_f.grid(row=len(JOINT_NAMES_CTRL)+1, column=0,
                   columnspan=5, pady=(8,0), sticky='ew')
        tk.Label(dur_f, text='Duración del movimiento:',
                 font=('Helvetica', 10)).pack(side='left')
        self.duration_var = tk.DoubleVar(value=1.5)
        ttk.Scale(dur_f, from_=0.2, to=5.0, orient='horizontal',
                  variable=self.duration_var, length=150,
                  command=lambda v: self._update_dur_label()
                  ).pack(side='left', padx=6)
        self.dur_label = tk.Label(dur_f, text='1.5 s',
                                  font=('Helvetica', 10), width=5)
        self.dur_label.pack(side='left')

        # Botones
        btn_f = tk.Frame(left)
        btn_f.grid(row=len(JOINT_NAMES_CTRL)+2, column=0,
                   columnspan=5, pady=(12,4), sticky='ew')

        self.send_btn = tk.Button(
            btn_f, text='▶  Enviar a Gazebo',
            font=('Helvetica', 11, 'bold'),
            bg='#2e7d32', fg='white',
            activebackground='#1b5e20', activeforeground='white',
            padx=14, pady=6, relief='flat',
            command=self._send_to_gazebo)
        self.send_btn.pack(side='left', padx=(0,8))

        tk.Button(btn_f, text='⟳  Cero',
                  font=('Helvetica', 10), padx=10, pady=6,
                  relief='flat', bg='#e0e0e0',
                  command=self._reset_all).pack(side='left')

        self.status_var = tk.StringVar(value='Listo')
        tk.Label(left, textvariable=self.status_var,
                 font=('Helvetica', 9), fg='gray'
                 ).grid(row=len(JOINT_NAMES_CTRL)+3, column=0,
                        columnspan=5, pady=(4,0))

        # ── Panel derecho ────────────────────────────────────────
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky='n')

        pos_frame = tk.LabelFrame(right, text='Posición del efector final (base_link)',
                                  font=('Helvetica', 10, 'bold'),
                                  padx=12, pady=8)
        pos_frame.pack(fill='x', pady=(0,8))

        self.pos_vars = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            tk.Label(pos_frame, text=f'{axis} =',
                     font=('Helvetica', 12, 'bold'), width=3
                     ).grid(row=i, column=0)
            var = tk.StringVar(value='0.0000 m')
            tk.Label(pos_frame, textvariable=var,
                     font=('Courier', 12), width=12, anchor='w'
                     ).grid(row=i, column=1, padx=(4,0))
            self.pos_vars[axis] = var

        mth_frame = tk.LabelFrame(right,
                                  text='Matrices de Transformación Homogénea',
                                  font=('Helvetica', 10, 'bold'),
                                  padx=8, pady=6)
        mth_frame.pack(fill='both', expand=True)

        self.mth_text = scrolledtext.ScrolledText(
            mth_frame, width=54, height=30,
            font=('Courier', 9), state='disabled',
            bg='#1e1e1e', fg='#d4d4d4')
        self.mth_text.pack()

    def _on_slider(self, var):
        # Actualizar todos los labels
        for joint, sv in self.sliders.items():
            self.angle_labels[joint].config(text=f'{sv.get():+.1f}°')
        self._update_fk()

    def _update_dur_label(self):
        self.dur_label.config(text=f'{self.duration_var.get():.1f} s')

    def _reset_all(self):
        for j, var in self.sliders.items():
            var.set(0.0)
            self.angle_labels[j].config(text='  0.0°')
        self._update_fk()

    def _read_real_state(self):
        real = self.node.get_real_positions_deg()
        for j in JOINT_NAMES_CTRL:
            deg = real.get(j, 0.0)
            self.sliders[j].set(deg)
            self.angle_labels[j].config(text=f'{deg:+.1f}°')
        self._update_fk()
        self.status_var.set('Estado real cargado desde /joint_states')

    def _update_fk(self):
        j1 = self.sliders['joint_1'].get()
        j2 = self.sliders['joint_2'].get()
        j3 = self.sliders['joint_3'].get()

        matrices, x, y, z = forward_kinematics(j1, j2, j3)

        self.pos_vars['x'].set(f'{x:+.4f} m')
        self.pos_vars['y'].set(f'{y:+.4f} m')
        self.pos_vars['z'].set(f'{z:+.4f} m')

        text_parts = [mat_str(k, v) for k, v in matrices.items()]
        full_text = '\n\n'.join(text_parts)

        self.mth_text.config(state='normal')
        self.mth_text.delete('1.0', tk.END)
        self.mth_text.insert(tk.END, full_text)
        self.mth_text.config(state='disabled')

        self.node.publish_fk(matrices, x, y, z)

    def _send_to_gazebo(self):
        j1  = self.sliders['joint_1'].get()
        j2  = self.sliders['joint_2'].get()
        j3  = self.sliders['joint_3'].get()
        dur = self.duration_var.get()

        self.send_btn.config(state='disabled', text='Enviando…')
        self.status_var.set('Enviando goal al controlador…')

        self.node.send_to_gazebo(j1, j2, j3, dur,
                                 self._goal_response_cb,
                                 self._goal_result_cb)

    def _goal_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.status_var.set('⚠ Goal rechazado')
            self._re_enable_btn()
            return
        self.status_var.set('Ejecutando movimiento…')
        gh.get_result_async().add_done_callback(self._goal_result_cb)

    def _goal_result_cb(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.status_var.set('✓ Posición alcanzada')
        else:
            self.status_var.set(f'⚠ Error: {result.error_string}')
        self._re_enable_btn()

    def _re_enable_btn(self):
        self.send_btn.config(state='normal', text='▶  Enviar a Gazebo')

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