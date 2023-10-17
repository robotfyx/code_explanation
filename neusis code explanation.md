Neusis代码解读，代码来自[neusis](https://github.com/rpl-cmu/neusis)

---

1.熟悉配置文件。该文件夹中，confs文件夹下是若干待重建场景的配置文件：<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/1.1.png)<br />我们以14deg_planeFull.conf为例，来看一下配置文件中都包含哪些内容：<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/1.2.png)<br />可见其中包含基本参数（文件名等）、训练相关参数、网格相关参数、模型参数（包含SDF、variance_network?、渲染网络三个神经网络与一个渲染器）

---

2.加载数据集<br />在该数据集中，包含一个Data文件夹与一个Config.json文件，Data文件夹中为若干.pkl文件，Config.json文件中是有关该传感器（本代码中为成像声呐）的相关参数。<br />使用load_data.py来加载数据集，返回值为一个字典，包含内容如下：<br />images：图片<br />images_no_noise：不含噪声图片（是一个空集）<br />sensor_poses：传感器位姿，是一个4*4矩阵，可将传感器坐标系转换为世界坐标系<br />min_range/max_range：最近、最远距离<br />hfov：声呐的方位角范围（θ的范围），弧度制<br />vfov：声呐的仰角范围（φ的范围），弧度制

---

3.下面从train()函数开始分析<br />训练流程是：从start_iter（0）到end_iter（300000）（间隔图片张数），每次循环中，将所有图片遍历一遍<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/3.1.png)<br />打乱i_train，三种损失初始值为0，进入循环<br />获取目标图像target、位姿矩阵pose<br />`coords, target = self.getRandomImgCoordsByPercentage(target)`通过百分比获得坐标点<br />返回的coords为若干（像素值不为0的点以及N_rand个随机点）像素值的索引，target为移入GPU的tensor（在生成coords过程中所需要的del_coords删去了0-64行的点，这步的作用没有搞明白）

---

4.
```
n_pixels = len(coords)
rays_d, dphi, r, rs, pts, dists = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                        self.randomize_points, self.device, self.cube_center)
```
下面使用get_arcs函数，获得每一点处的圆弧上的采样点<br />get_arcs函数：<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/4.1.png)<br />传入参数如下：<br />H、W：图像的高度、宽度<br />phi_min/phi_max：φ的最小值与最大值，定义式如下：
```
self.phi_min = -self.data["vfov"]/2
self.phi_max = self.data["vfov"]/2
```
r_min/r_max：最近、最远距离
```
self.r_min = self.data["min_range"]
self.r_max = self.data["max_range"]
```
c2w：位姿矩阵，即为pose<br />n_selected_px：选取的坐标点个数，即n_pixels = len(coords)<br />arc_n_samples：每条弧上选取点个数，.conf中获取，值为10<br />ray_n_samples：每条射线上选取点个数，.conf中获取，值为64<br />hfov：即从数据集中获得的hfov<br />px：采样点集合，即coords<br />r_increments：每个ri对应的距离值
```
r_increments = []
self.sonar_resolution = (self.r_max-self.r_min)/self.H
for i in range(self.H):
    r_increments.append(i*self.sonar_resolution + self.r_min)

self.r_increments = torch.FloatTensor(r_increments).to(self.device)
```
randomize_points：是否随机采样，值为True<br />device：使用GPU<br />cube_center：重建时截取的立方体中心？
```
self.x_max = self.conf.get_float('mesh.x_max')
self.x_min = self.conf.get_float('mesh.x_min')          
self.y_max = self.conf.get_float('mesh.y_max')          
self.y_min = self.conf.get_float('mesh.y_min')
self.z_max = self.conf.get_float('mesh.z_max')
self.z_min = self.conf.get_float('mesh.z_min')

self.cube_center = torch.Tensor([(self.x_max + self.x_min)/2, (self.y_max + self.y_min)/2, (self.z_max + self.z_min)/2])
```
采样的方法如图：<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/4.2.png)<br />首先在每段弧上选取NA个点，再在以这些点为终点的射线上采样NR-1个点<br />返回值说明：<br />dirs:每条射线在世界坐标系中表示的方向（3维单位向量表示）<br />dphi：φ的微分值（`dphi = (phi_max - phi_min) / arc_n_samples`）<br />r：所有采样点的距离值`r = i*sonar_resolution + r_min`<br />rs：形状为n_selected_px ×arc_n_samples，为每个采样像素点的所有弧上点对应的距离值<br />pts_r_rand：将所有点换为笛卡尔坐标系表示后，将中心定为坐标原点的xyz坐标<br />dists：每条射线上前后点的距离值差分，dr

---

5.按照渲染公式进行渲染
```
render_out = self.renderer.render_sonar(rays_d, pts, dists, n_pixels, 
                                        self.arc_n_samples, self.ray_n_samples,
                                        cos_anneal_ratio=self.get_cos_anneal_ratio())
```
![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/5.1.png)<br />传入参数：rays_d, pts, dists为由get_arcs函数得到的方向、rs距离值、距离值差分，<br />sdf_network等三个网络<br />n_pixels：采样像素点个数<br />arc_n_samples：每条弧上选取点个数，.conf中获取，值为10<br />ray_n_samples：每条射线上选取点个数，.conf中获取，值为64<br />cos_anneal_ratio：由以下函数获得：
```
def get_cos_anneal_ratio(self):
    if self.anneal_end == 0.0:
        return 1.0
    else:
        return np.min([1.0, self.iter_step / self.anneal_end])
```
在训练迭代次数小于anneal_end（50000）之前，值为self.iter_step / self.anneal_end，之后为1<br />将参数传入render_core_sonar函数进行渲染<br />返回值：<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/5.2.png)

---

6.计算损失，反向更新<br />![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/6.png)

---

7.保存权重、加载权重、更新学习率、验证重建网络结构函数![image.png](https://github.com/robotfyx/code_explanation/raw/main/imgs/Neusis/7.png)
