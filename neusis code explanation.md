Neusis代码解读，代码来自[neusis](https://github.com/rpl-cmu/neusis)

---

1.熟悉配置文件。该文件夹中，confs文件夹下是若干待重建场景的配置文件：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697464287882-eeabc1aa-cec3-4cb0-a9cc-badb1fab1517.png#averageHue=%23212e3a&clientId=u050d20bf-db2a-4&from=paste&height=137&id=u18024e49&originHeight=171&originWidth=274&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=8448&status=done&style=none&taskId=ufe93ce7f-be41-482d-be59-6289250128c&title=&width=219.2)<br />我们以14deg_planeFull.conf为例，来看一下配置文件中都包含哪些内容：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697464451608-4d752ca8-c379-4c18-b912-73766b9ac414.png#averageHue=%231a2530&clientId=u050d20bf-db2a-4&from=paste&height=1436&id=ua9441f1a&originHeight=1795&originWidth=668&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=103679&status=done&style=none&taskId=u4d4ac681-17d9-4949-8be1-4356527830e&title=&width=534.4)<br />可见其中包含基本参数（文件名等）、训练相关参数、网格相关参数、模型参数（包含SDF、variance_network?、渲染网络三个神经网络与一个渲染器）

---

2.加载数据集<br />在该数据集中，包含一个Data文件夹与一个Config.json文件，Data文件夹中为若干.pkl文件，Config.json文件中是有关该传感器（本代码中为成像声呐）的相关参数。<br />使用load_data.py来加载数据集，返回值为一个字典，包含内容如下：<br />images：图片<br />images_no_noise：不含噪声图片（是一个空集）<br />sensor_poses：传感器位姿，是一个4*4矩阵，可将传感器坐标系转换为世界坐标系<br />min_range/max_range：最近、最远距离<br />hfov：声呐的方位角范围（θ的范围），弧度制<br />vfov：声呐的仰角范围（φ的范围），弧度制

---

3.下面从train()函数开始分析<br />训练流程是：从start_iter（0）到end_iter（300000）（间隔图片张数），每次循环中，将所有图片遍历一遍<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697465567933-e8268677-e164-4e03-96fd-f6d031986682.png#averageHue=%231a2530&clientId=u050d20bf-db2a-4&from=paste&height=263&id=u95052de8&originHeight=329&originWidth=811&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=26431&status=done&style=none&taskId=ufb7698ed-37ac-4968-8a48-61724f42be4&title=&width=648.8)<br />打乱i_train，三种损失初始值为0，进入循环<br />获取目标图像target、位姿矩阵pose<br />`coords, target = self.getRandomImgCoordsByPercentage(target)`通过百分比获得坐标点<br />返回的coords为若干（像素值不为0的点以及N_rand个随机点）像素值的索引，target为移入GPU的tensor（在生成coords过程中所需要的del_coords删去了0-64行的点，这步的作用没有搞明白）

---

4.
```
n_pixels = len(coords)
rays_d, dphi, r, rs, pts, dists = get_arcs(self.H, self.W, self.phi_min, self.phi_max, self.r_min, self.r_max,  torch.Tensor(pose), n_pixels,
                                        self.arc_n_samples, self.ray_n_samples, self.hfov, coords, self.r_increments, 
                                        self.randomize_points, self.device, self.cube_center)
```
下面使用get_arcs函数，获得每一点处的圆弧上的采样点<br />get_arcs函数：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697467195547-c361cc07-af23-4eca-bd0c-e06a8d666872.png#averageHue=%2325313d&clientId=u050d20bf-db2a-4&from=paste&height=49&id=u7a6b0734&originHeight=61&originWidth=915&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7861&status=done&style=none&taskId=u25c64bf1-f341-46c4-acf6-37b4de4625e&title=&width=732)<br />传入参数如下：<br />H、W：图像的高度、宽度<br />phi_min/phi_max：φ的最小值与最大值，定义式如下：
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
采样的方法如图：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697542975054-89ffcee5-66a4-4436-93a0-d4c79b05423f.png#averageHue=%23fbf9f8&clientId=u56171874-6b9e-4&from=paste&height=370&id=ufa9704f3&originHeight=463&originWidth=619&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=139442&status=done&style=none&taskId=u2440ac6d-3746-4222-9c3e-b3b26af0987&title=&width=495.2)<br />首先在每段弧上选取NA个点，再在以这些点为终点的射线上采样NR-1个点<br />返回值说明：<br />dirs:每条射线在世界坐标系中表示的方向（3维单位向量表示）<br />dphi：φ的微分值（`dphi = (phi_max - phi_min) / arc_n_samples`）<br />r：所有采样点的距离值`r = i*sonar_resolution + r_min`<br />rs：形状为n_selected_px ×arc_n_samples，为每个采样像素点的所有弧上点对应的距离值<br />pts_r_rand：将所有点换为笛卡尔坐标系表示后，将中心定为坐标原点的xyz坐标<br />dists：每条射线上前后点的距离值差分，dr

---

5.按照渲染公式进行渲染
```
render_out = self.renderer.render_sonar(rays_d, pts, dists, n_pixels, 
                                        self.arc_n_samples, self.ray_n_samples,
                                        cos_anneal_ratio=self.get_cos_anneal_ratio())
```
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697545875072-c6c03f2a-5a50-47fa-8f94-ea190d9f79e0.png#averageHue=%231a252f&clientId=u56171874-6b9e-4&from=paste&height=516&id=uf8993ceb&originHeight=645&originWidth=822&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=53938&status=done&style=none&taskId=u40ff4b44-af04-43a3-9dc7-f9b06c0f59f&title=&width=657.6)<br />传入参数：rays_d, pts, dists为由get_arcs函数得到的方向、rs距离值、距离值差分，<br />sdf_network等三个网络<br />n_pixels：采样像素点个数<br />arc_n_samples：每条弧上选取点个数，.conf中获取，值为10<br />ray_n_samples：每条射线上选取点个数，.conf中获取，值为64<br />cos_anneal_ratio：由以下函数获得：
```
def get_cos_anneal_ratio(self):
    if self.anneal_end == 0.0:
        return 1.0
    else:
        return np.min([1.0, self.iter_step / self.anneal_end])
```
在训练迭代次数小于anneal_end（50000）之前，值为self.iter_step / self.anneal_end，之后为1<br />将参数传入render_core_sonar函数进行渲染<br />返回值：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697547192439-856df641-9002-4259-9f1f-ae3c4c03e3e4.png#averageHue=%231a2530&clientId=u56171874-6b9e-4&from=paste&height=206&id=u193592f4&originHeight=258&originWidth=604&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18458&status=done&style=none&taskId=ueaf0e451-97e3-476d-99e1-7d16fa33946&title=&width=483.2)

---

6.计算损失，反向更新<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697547293945-b8cafcdd-a761-4fab-8bd0-82f2ffe6e1f5.png#averageHue=%231a2530&clientId=u56171874-6b9e-4&from=paste&height=513&id=ua5d6a558&originHeight=641&originWidth=1017&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=65294&status=done&style=none&taskId=u9f5229f0-c53c-4552-845c-b2a9179a1b9&title=&width=813.6)

---

7.保存权重、加载权重、更新学习率、验证重建网络结构函数![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697547390453-e93b4d85-2662-4642-aba4-ccb70825cd45.png#averageHue=%231a2631&clientId=u56171874-6b9e-4&from=paste&height=838&id=ud5ae1b7e&originHeight=1047&originWidth=1121&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=132012&status=done&style=none&taskId=u81baf2f7-546f-4095-a388-b97d930f277&title=&width=896.8)
