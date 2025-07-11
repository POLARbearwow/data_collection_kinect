

## 1. 像素坐标 → 相机坐标系射线

设相机内参  

$$
K=\begin{bmatrix}
f_x & 0   & c_x\\
0   & f_y & c_y\\
0   & 0   & 1
\end{bmatrix},
$$

像素点 $(u,v)$ 归一化得  

$$
\tilde{\mathbf d}=
\begin{bmatrix}
(u-c_x)/f_x\\
(v-c_y)/f_y\\
1
\end{bmatrix},
\qquad
\mathbf d_{\text{cam}}
=\frac{\tilde{\mathbf d}}{\|\tilde{\mathbf d}\|}.
$$

---

## 2. 相机坐标系 → ArUco（世界）坐标系

`estimatePoseSingleMarkers` 返回姿态  

$$
\mathbf p_{\text{cam}}=R\,\mathbf p_{\text{aruco}}+\mathbf t.
$$

- 相机光心在世界系中的坐标  

$$
\boxed{\mathbf c_{\text{aruco}}=-R^{\mathsf T}\mathbf t}.
$$

- 射线方向转到世界系  

$$
\boxed{\mathbf d_{\text{aruco}}=R^{\mathsf T}\mathbf d_{\text{cam}} }.
$$

- 射线参数方程  

$$
\boxed{\mathbf p(s)=\mathbf c_{\text{aruco}}+s\,\mathbf d_{\text{aruco}},\;s\ge0}.
$$

---

## 3. 射线与已知平面求交

平面 $\Pi$ 由法向量 $\mathbf n$ 与平面点 $\mathbf p_0$ 描述：  

$$
\mathbf n\cdot(\mathbf p-\mathbf p_0)=0.
$$

将射线代入，求参数  

$$
\boxed{s=\dfrac{\mathbf n\cdot(\mathbf p_0-\mathbf c_{\text{aruco}})}
               {\mathbf n\cdot\mathbf d_{\text{aruco}}}}.
$$

若分母近 0 ⇒ 平行；若 $s<0$ ⇒ 交点在相机后；两者均无效。  
有效时交点为  

$$
\boxed{\mathbf p_\ast=\mathbf c_{\text{aruco}}+s\,\mathbf d_{\text{aruco}} }.
$$



---
