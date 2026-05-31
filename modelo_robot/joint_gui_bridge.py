#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import threading
import tkinter as tk
from tkinter import ttk

CONTROLLED_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4']
JOINT_LIMITS = {
    'joint_1': (-1.5708, 1.5708),
    'joint_2': (-1.5708, 1.5708),
    'joint_3': (-1.5708, 1.5708),
    'joint_4': (-1.5708, 1.5708),
}

class JointControlGUI(Node):
    def __init__(self):
        super().__init__('joint_control_gui')

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        self._action_client.wait_for_server()
        self.get_logger().info('Conectado al controlador. GUI lista.')

        self._build_gui()

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title('Control de Joints — modelo_robot')
        self.root.resizable(False, False)

        self.sliders = {}
        self.value_labels = {}

        # ── Título ──────────────────────────────────────────────
        title = tk.Label(
            self.root,
            text='modelo_robot — Control de articulaciones',
            font=('Helvetica', 13, 'bold'),
            pady=10
        )
        title.pack()

        frame = tk.Frame(self.root, padx=20, pady=10)
        frame.pack()

        # ── Sliders ─────────────────────────────────────────────
        for row, joint in enumerate(CONTROLLED_JOINTS):
            lo, hi = JOINT_LIMITS[joint]

            # Nombre del joint
            tk.Label(frame, text=joint, font=('Helvetica', 11), width=10,
                     anchor='w').grid(row=row, column=0, padx=(0, 10), pady=8)

            # Límite inferior
            tk.Label(frame, text=f'{lo:.2f}', font=('Helvetica', 9),
                     fg='gray').grid(row=row, column=1)

            # Slider
            var = tk.DoubleVar(value=0.0)
            slider = ttk.Scale(
                frame,
                from_=lo,
                to=hi,
                orient='horizontal',
                variable=var,
                length=300,
                command=lambda val, j=joint, v=var: self._on_slider_move(j, v)
            )
            slider.grid(row=row, column=2, padx=8)
            self.sliders[joint] = var

            # Límite superior
            tk.Label(frame, text=f'{hi:.2f}', font=('Helvetica', 9),
                     fg='gray').grid(row=row, column=3)

            # Valor actual
            lbl = tk.Label(frame, text='0.000 rad', font=('Helvetica', 10),
                           width=10, anchor='e')
            lbl.grid(row=row, column=4, padx=(10, 0))
            self.value_labels[joint] = lbl

        # ── Tiempo de ejecución ──────────────────────────────────
        time_frame = tk.Frame(self.root, padx=20)
        time_frame.pack(pady=(5, 0))

        tk.Label(time_frame, text='Duración del movimiento:',
                 font=('Helvetica', 10)).pack(side='left')

        self.duration_var = tk.DoubleVar(value=1.0)
        duration_slider = ttk.Scale(
            time_frame,
            from_=0.1,
            to=5.0,
            orient='horizontal',
            variable=self.duration_var,
            length=200,
            command=lambda v: self._update_duration_label()
        )
        duration_slider.pack(side='left', padx=8)

        self.duration_label = tk.Label(time_frame, text='1.0 s',
                                       font=('Helvetica', 10), width=6)
        self.duration_label.pack(side='left')

        # ── Botones ──────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, pady=15)
        btn_frame.pack()

        self.send_btn = tk.Button(
            btn_frame,
            text='▶  Enviar posición',
            font=('Helvetica', 12, 'bold'),
            bg='#2e7d32',
            fg='white',
            activebackground='#1b5e20',
            activeforeground='white',
            padx=20,
            pady=8,
            relief='flat',
            command=self._send_goal
        )
        self.send_btn.pack(side='left', padx=10)

        tk.Button(
            btn_frame,
            text='⟳  Resetear a cero',
            font=('Helvetica', 11),
            padx=15,
            pady=8,
            relief='flat',
            bg='#e0e0e0',
            command=self._reset_all
        ).pack(side='left', padx=10)

        # ── Estado ───────────────────────────────────────────────
        self.status_var = tk.StringVar(value='Listo')
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=('Helvetica', 10),
            fg='gray',
            pady=6
        )
        status_bar.pack()

    def _on_slider_move(self, joint, var):
        val = var.get()
        self.value_labels[joint].config(text=f'{val:+.3f} rad')

    def _update_duration_label(self):
        self.duration_label.config(text=f'{self.duration_var.get():.1f} s')

    def _reset_all(self):
        for joint, var in self.sliders.items():
            var.set(0.0)
            self.value_labels[joint].config(text=' 0.000 rad')

    def _send_goal(self):
        positions = [self.sliders[j].get() for j in CONTROLLED_JOINTS]
        duration_sec = self.duration_var.get()

        self.send_btn.config(state='disabled', text='Enviando…')
        self.status_var.set('Enviando goal al controlador…')

        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        trajectory = JointTrajectory()
        trajectory.joint_names = CONTROLLED_JOINTS
        trajectory.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        future = self._action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.status_var.set('⚠ Goal rechazado por el controlador')
            self._re_enable_button()
            return
        self.status_var.set('Ejecutando movimiento…')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result().result
        if result.error_code == 0:
            self.status_var.set('✓ Posición alcanzada')
        else:
            self.status_var.set(f'⚠ Error: {result.error_string}')
        self._re_enable_button()

    def _re_enable_button(self):
        self.send_btn.config(state='normal', text='▶  Enviar posición')

    def run(self):
        self.root.mainloop()


def main(args=None):
    rclpy.init(args=args)
    node = JointControlGUI()

    # ROS 2 spin en hilo separado para no bloquear tkinter
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    node.run()  # Bloquea hasta cerrar la ventana

    rclpy.shutdown()

if __name__ == '__main__':
    main()