from vpython import canvas, vector, box, cylinder, color, rate, curve, sphere, label
import numpy as np


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]])

    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])

    return Rz @ Ry @ Rx


class UAV3D:

    def __init__(self, col=color.white, pos=vector(0,1.0,0), scale=0.3):
        self.scale = scale
        self.pos = pos

        # ===================== 机身 ============================
        self.body = cylinder(
            pos=pos,
            axis=vector(0,0,scale*0.25),
            radius=scale*0.25,
            color=col
        )

        # ===================== 前向灯（区别机头） =====================
        self.front_led = sphere(
            pos=pos + vector(scale*0.3, 0, 0),
            radius=scale*0.08,
            color=color.red
        )

        # ===================== 三轴 ============================
        axis_len = scale * 0.6
        self.x_axis = cylinder(pos=pos, axis=vector(axis_len,0,0), radius=scale*0.03, color=color.red)
        self.y_axis = cylinder(pos=pos, axis=vector(0,axis_len,0), radius=scale*0.03, color=color.green)
        self.z_axis = cylinder(pos=pos, axis=vector(0,0,axis_len), radius=scale*0.03, color=color.blue)

        self.objects = [self.body, self.front_led,
                        self.x_axis, self.y_axis, self.z_axis]

        # ===================== 四个机臂（更真实） =====================
        arm_len = scale * 1.0
        arm_rad = scale * 0.05

        self.arms = []
        arm_dirs = [
            vector(1, 0, 1),  # front-right
            vector(-1, 0, 1), # front-left
            vector(1, 0, -1), # back-right
            vector(-1,0,-1)   # back-left
        ]

        for d in arm_dirs:
            d_hat = d.norm()
            arm = cylinder(
                pos=pos,
                axis=d_hat * arm_len,
                radius=arm_rad,
                color=color.gray(0.7)
            )
            self.arms.append(arm)

        # ===================== 电机 + 旋翼（随姿态倾斜） =====================
        self.rotors = []
        rotor_r = scale * 0.28

        for i in range(4):
            rotor = cylinder(
                pos=pos,
                axis=vector(0,0,0.02),
                radius=rotor_r,
                color=color.gray(0.9)
            )
            self.rotors.append(rotor)

        # rotor local offset
        self.rotor_local = np.array([
            [ scale*0.7, 0,  scale*0.7],
            [-scale*0.7, 0,  scale*0.7],
            [ scale*0.7, 0, -scale*0.7],
            [-scale*0.7, 0, -scale*0.7],
        ])

        # ===================== 轨迹 =====================
        self.trail = curve(color=col, radius=scale*0.015)
        self.trail.append(pos)


    # ================================================================
    def set_position(self, pos):
        self.pos = pos
        for obj in self.objects + self.arms + self.rotors:
            obj.pos = pos
        self.trail.append(pos)


    # ================================================================
    def set_rotation(self, R):
        x_dir = vector(R[0,0], R[1,0], R[2,0])
        y_dir = vector(R[0,1], R[1,1], R[2,1])
        z_dir = vector(R[0,2], R[1,2], R[2,2])

        # 机身
        self.body.axis = z_dir * (self.scale * 0.25)

        # 前灯跟随机头
        self.front_led.pos = self.pos + x_dir * (self.scale * 0.3)

        # 三轴
        axis_len = self.scale * 0.6
        self.x_axis.axis = x_dir * axis_len
        self.y_axis.axis = y_dir * axis_len
        self.z_axis.axis = z_dir * axis_len

        # 四个机臂
        arm_dirs = [
            (x_dir + z_dir).norm(),
            (-x_dir + z_dir).norm(),
            (x_dir - z_dir).norm(),
            (-x_dir - z_dir).norm()
        ]

        for arm, d in zip(self.arms, arm_dirs):
            arm.axis = d * self.scale * 1.0

        # 旋翼随姿态倾斜
        for i, off in enumerate(self.rotor_local):
            world_offset = R @ off
            self.rotors[i].pos = self.pos + vector(*world_offset)
            self.rotors[i].axis = y_dir * 0.02


    # ================================================================
    def spin_rotors(self):
        for rotor in self.rotors:
            rotor.rotate(angle=1.0, axis=rotor.axis)  # 更真实的旋翼旋转



def animate_uavs(pursuer_att, evader_att,
                 refresh_rate=0.02,
                 capture_tol=0.03,
                 success_frames_needed=5):
    """
    pursuer_att: (T, N, 3) -> [roll, pitch, yaw] (rad)
    evader_att : (T, 3)
    capture_tol: 所有追击者姿态误差范数 < capture_tol 时，认为追捕成功
    success_frames_needed: 连续多少帧都满足条件，才判定成功，防止抖动
    """

    T, N, _ = pursuer_att.shape

    # ================== 画布 / 相机设置（更舒服的“天空+地面”风格） ==================
    scene = canvas(
        width=1200, height=700,
        background=vector(0.85, 0.90, 1.0),   # 浅蓝色天空
        forward=vector(-0.6, -0.7, -0.3),
        up=vector(0, 1, 0)
    )
    scene.userspin = True
    scene.userzoom = True
    scene.userpan  = True
    scene.range = 12
    scene.caption = "Multi-QUAV Pursuit-Evasion (Attitude Game)\n"

    # —— 地面：浅绿色平台 + 稍微淡一点的网格线 —— #
    ground = box(
        pos=vector(0, -2.5, 0),
        size=vector(40, 0.2, 40),
        color=vector(0.75, 0.85, 0.75),
        opacity=1.0
    )

    # 稍微淡一点的网格
    grid_color = color.gray(0.7)
    for x in range(-20, 21, 2):
        curve(pos=[vector(x, -2.4, -20), vector(x, -2.4, 20)],
              color=grid_color, radius=0.01)
    for z in range(-20, 21, 2):
        curve(pos=[vector(-20, -2.4, z), vector(20, -2.4, z)],
              color=grid_color, radius=0.01)

    # ================== 初始化 UAV 模型 ==================
    evader = UAV3D(col=color.yellow, pos=vector(0, 1.0, 0), scale=0.4)

    pursuers = []
    for i in range(N):
        pursuers.append(UAV3D(
            col=color.gray(0.9),
            pos=vector(0, 1.0, 0),
            scale=0.3
        ))

    # 姿态标签（可留可删，这里只给逃避者一个）
    evader_label = label(pos=evader.pos + vector(0, 0.8, 0),
                         text="",
                         height=14, box=False, color=color.yellow)

    # ================== 轨迹 / 位置运动设计 ==================
    # 不再“绕圈圈”：追击者只做简单的“直线靠近”，主要用来辅助视觉
    # 逃避者在前方缓慢飞行
    def evader_pos(t):
        t_sec = t * refresh_rate
        return vector(
            0.2 * t_sec,                          # 向前匀速飞一点
            1.2 + 0.1 * np.sin(0.5 * t_sec),      # 轻微上下起伏
            0.5 * np.sin(0.3 * t_sec)             # 左右小幅摆动
        )

    # 追击者初始相对位置：一开始分散在逃避者周围
    init_offsets = []
    radius0 = 6.0
    for i in range(N):
        ang = 2 * np.pi * i / N
        # 初始相对于逃避者的位置
        off = vector(radius0 * np.cos(ang),
                     0.5 * (i - (N-1)/2),
                     radius0 * np.sin(ang))
        init_offsets.append(off)

    # 追击者运动：沿直线收缩到逃避者附近，不再绕圈圈
    def pursuer_pos(t, i):
        center = evader_pos(t)
        # 收缩系数：0 时还在初始位置，1 时完全贴近逃避者
        # 这里选择一个在 20 秒左右基本贴近的速度
        t_sec = t * refresh_rate
        shrink = min(1.0, t_sec / 20.0)  # 约 20 秒收缩完
        off = init_offsets[i] * (1.0 - shrink)
        return center + off

    # ================== 相机平滑跟随逃避者 ==================
    cam_center = vector(0, 0, 0)
    follow_strength = 0.05

    # ================== 追捕成功逻辑 ==================
    captured = False
    success_counter = 0   # 连续满足“所有误差很小”的帧数

    # 画面中央大字（先创建，开始时隐藏）
    success_label = label(
        pos=vector(0, 3, 0),
        text="",
        height=40,
        color=color.red,
        box=False,
        opacity=0,
        billboard=True,
        align='center'
    )

    # ================== 主循环 ==================
    for t in range(T):
        rate(int(1 / refresh_rate))

        # 当前仿真时间（秒）
        t_sec = t * refresh_rate

        # ---- 更新逃避者姿态 ----
        roll_s, pitch_s, yaw_s = evader_att[t]
        R_e = rotation_matrix(roll_s, pitch_s, yaw_s)
        pos_e = evader_pos(t)

        evader.set_position(pos_e)
        evader.set_rotation(R_e)
        evader.spin_rotors()

        # 逃避者姿态显示（可选）
        rs_deg, ps_deg, ys_deg = map(np.degrees, [roll_s, pitch_s, yaw_s])
        evader_label.pos = pos_e + vector(0, 0.8, 0)
        evader_label.text = f"E  R={rs_deg:5.1f}°  P={ps_deg:5.1f}°  Y={ys_deg:5.1f}°"

        # ---- 更新追击者，并计算姿态误差 ----
        max_err = 0.0
        for i in range(N):
            roll_p, pitch_p, yaw_p = pursuer_att[t, i]
            R_p = rotation_matrix(roll_p, pitch_p, yaw_p)
            pos_p = pursuer_pos(t, i)

            pursuers[i].set_position(pos_p)
            pursuers[i].set_rotation(R_p)
            pursuers[i].spin_rotors()

            # 姿态误差范数
            err_vec = np.array([roll_p - roll_s,
                                pitch_p - pitch_s,
                                yaw_p - yaw_s])
            err = np.linalg.norm(err_vec)
            if err > max_err:
                max_err = err

        # ---- 检查是否“所有追击者都几乎对齐” ----
        if max_err < capture_tol:
            success_counter += 1
        else:
            success_counter = 0

        # 连续 success_frames_needed 帧误差都很小 -> 追捕成功
        if success_counter >= success_frames_needed and (not captured):
            captured = True

            # 所有追击者变成绿色，突出成功
            for i in range(N):
                pursuers[i].body.color = color.green
                for obj in pursuers[i].objects:
                    obj.color = color.green

            # 屏幕中央显示“追捕成功”
            success_label.text = "追捕成功"
            success_label.opacity = 0     # 文字本身不需要背景框

            scene.caption += f"\n追捕成功：t ≈ {t_sec:.1f} s, max attitude error = {max_err:.4f} rad"

            # 停止动画：直接跳出循环
            break

        # ---- 相机平滑跟随逃避者 ----
        cam_center = cam_center * (1 - follow_strength) + pos_e * follow_strength
        scene.center = cam_center

    print("Animation finished. captured =", captured)


