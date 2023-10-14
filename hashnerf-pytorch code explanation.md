Hashnerf-pytorch代码解读，代码来自[Hashnerf-pytorch](https://github.com/yashbhalgat/HashNeRF-pytorch)

---

从train()函数开始<br />首先定义了一个K=None，这个K是相机内参矩阵，用于将三维世界的坐标映射为二维图像，在之后会用到

---

1.读取数据集，这里由于数据集的dataset_type有llff/blender/deepvoxels等多种，因此分别有对应的多个读取数据的函数，例如：

```
if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test, bounding_box = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        args.bounding_box = bounding_box
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
```

```
elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
        args.bounding_box = bounding_box
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
```

等，有关数据集格式不是我们研究的重点，只要知道在这里都得到了什么值即可。<br />返回值说明：<br />images：图片集合，（N，H，W，3），N为数据集图像张数，H、W为图片宽度、长度像素数<br />poses：这是一个包含所有相机位姿的数组，每个位姿对应一个矩阵，描述了相机在世界坐标系中的位置和方向<br />render_poses：这是一个包含用于渲染的相机位姿的数组。这些位姿通常是沿着预定义路径. (例如螺旋路径)生成的，用于在训练过程中或在训练后生成新的视图<br />hwf：图像的高度、长度、相机焦距<br />near/far：用于表示场景的范围Bounds（bds），是该相机视角下场景点离相机中心最近(near)和最远(far)的距离<br />i_train/i_test/i_val：划分的训练、测试、验证集下标<br />bounding_box：场景中物体最近端与最远端顶点坐标，（tensor1，tensor2），每个tensor包含三个坐标（xyz）

---

2.调整相机参数

```
# Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
```

这部分基于第1步的结果重新写了一下hwf，并且给出了相机内参矩阵K的定义式

---

3.保存config部分，建了args.txt和config.txt用于保存参数和配置

---

4.重点，创建一个nerf model<br />![image.png](E:\code%20explanation\imgs\hashnerf\4.1.png)<br />所以我们来看一下create_nerf函数都干了什么<br />`embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)`，首先又进入了get_embedder函数。传入的参数：multires，对应的是传统的nerf的位置编码中的参数L，这里我们用的是Hashnerf，这个参数没有什么用；args，相应的参数配置；i_embed，给出的解释是help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical'，是用于选择编码方式的，默认是1，即哈希编码（代码在编码方向时，选择了2）。<br />下面来看get_embedder函数

```
def get_embedder(multires, args, i=0):
    if i == -1:
        return nn.Identity(), 3
    elif i==0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i==1:
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif i==2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    return embed, out_dim
```

其有两个返回值：embed，out_dim，先来看embed<br />embed是由三个编码类：Embedder、HashEmbedder、SHEncoder得到的，可以将输入进行编码，映射到高维空间。<br />例如，x是一个3D点的组合，形状为（B，3），embed(x）可以得到该位置对应的编码。~~（举例还存在cuda:0和cpu的问题，需要再细研究一下）~~<br />关于out_dim，在HashEmbedder中如下定义：<br />`self.out_dim = self.n_levels * self.n_features_per_level`表示输出的编码的个数，在论文中，分辨率级数为16，F=2，因此out_dim实际等于32<br />因此create_nerf第一行首先得到的embed_fn和input_ch分别为编码函数和数值32

```
if args.i_embed==1:
    # hashed embedding table
    embedding_params = list(embed_fn.parameters())

input_ch_views = 0
embeddirs_fn = None
```

embedding_param存储embed_fn的参数，input_ch_views=0，embeddirs_fn=None

```
if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)
```

use_viewdirs表示是否使用5D输入而不是3D输入，这里默认是是的，因此将view的两个参数通过SH编码<br />然后更改embeddirs_fn为编码方向函数，input_ch_view为输出维度（16）

```
output_ch = 5 if args.N_importance > 0 else 4
skips = [4]
```

N_importance表示沿一条射线额外精细采样的样本数，代码将其设为了0，因此output=4，skips=[4]

```
if args.i_embed==1:
    model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
```

创建了网络模型，这里应用了比原nerf更小的网络结构，网络结构为：输入为32位hash编码+16位sh编码，32位作为密度网络输入，32-64-16，取出第一个值作为密度，剩下15个值加上16位sh编码作为颜色网络输入，31-64-64-3，输出rgb值<br />然后定义了grad_vars存储model的参数值，model_fine=None<br />然后由于N_importance=0，model_fine和grad_vars保持原值

```
network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
```

这个函数的作用是将查询传递给神经网络，具体来说，它接收位置信息（inputs）、方向信息（viewdirs），nerf网络（network_fn）、两个编码函数（embed_fn、embeddirs_fn=embeddirs_fn），输出不透明度与颜色rgb值<br />下一步，定义了优化器optimizer，用的是RAdam<br />![loading-ag-207](E:\code%20explanation\imgs\hashnerf\4.2.png)<br />如果有现有模型权重，加载该权重

```
render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }
```

用于渲染训练的相关信息render_kwargs_train，并基于此定义了render_kwargs_test<br />由此完成了创建过程，返回值有：render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

---

5.<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697119511286-be07d34e-8203-4e1a-871d-3dd7dee9d107.png#averageHue=%231b2632&clientId=u6b71fa97-6e36-4&from=paste&height=136&id=u4e8c9bbe&originHeight=170&originWidth=344&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=7532&status=done&style=none&taskId=u890138b4-2517-425a-8ec7-258c026190b&title=&width=275.2)<br />global_step=start，render_kwargs_train和render_kwargs_test分别增加了两个参数：near和far

---

6.将render_poses移入GPU，如果只完成渲染过程不训练（已有训练好模型）（render_only）执行的部分<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697119609638-610f97b5-537f-4660-86be-69b14e8ad439.png#averageHue=%231b2631&clientId=u6b71fa97-6e36-4&from=paste&height=386&id=u665872dd&originHeight=483&originWidth=1134&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=52359&status=done&style=none&taskId=u566700df-e6ae-420b-9938-438d4b776cf&title=&width=907.2)

---

7.N_rand=32*32*4=4096，是每一步使用的射线数量，和论文保持一致<br />no_batching=True，则use_batching = not args.no_batching = False，表示每次只从一张图片中获取射线<br />将poses移入GPU<br />下面开始训练更新过程：<br />N_iters：训练代数，5000以上<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697120038338-227407f1-8a0e-4d77-b52d-c2a51fa25eb8.png#averageHue=%231a2530&clientId=u6b71fa97-6e36-4&from=paste&height=478&id=u5fa7405f&originHeight=597&originWidth=1068&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=69589&status=done&style=none&taskId=uf556699c-8726-4425-a9f3-d3e32054dc3&title=&width=854.4)<br />从i_train中随机抽取一张图片，target表示这张图片，pose表示该张图对应的相机位姿<br />之后获得rays_o，rays_d，表示每个像素的起点终点集合（每个的形状都为（H，W，3））<br />precrop_iters，help='number of steps to train on central crops'，=0<br />coords是所有像素坐标，形状为（H*W，2），即[[0,0],[0,1],……,[1,0],……]]<br />由此得到了随机抽取出的4096条射线的起终点rays_o，rays_d，从而得到batch_rays<br />target_s是这张图像中这些射线对应的像素点的颜色值，用与和预测值进行比较计算loss<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697120574084-dab101a3-bc60-41b1-ad1a-24cbdd8ef523.png#averageHue=%231a2631&clientId=u6b71fa97-6e36-4&from=paste&height=88&id=ud8a4fd27&originHeight=110&originWidth=713&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=9041&status=done&style=none&taskId=u2c192b6f-1e4e-4170-9b64-d573285ab1f&title=&width=570.4)<br />通过render函数，对这些射线进行渲染，得到颜色图、深度图、累计透射率分布图与其他信息<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697120689761-710099b8-63a6-4c07-a516-ca1d433a3202.png#averageHue=%231a2530&clientId=u6b71fa97-6e36-4&from=paste&height=658&id=u6e973ea6&originHeight=822&originWidth=1007&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=77230&status=done&style=none&taskId=ufaa17395-11fa-4fa9-86dd-efdfceb1139&title=&width=805.6)<br />计算损失，反向传播，更新权重

---

8.render()函数<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697122814845-1852a714-d8c7-4fc6-b33e-dea6157dd1f1.png#averageHue=%2319242e&clientId=u43ce8f52-26fb-4&from=paste&height=417&id=uf7052a02&originHeight=521&originWidth=832&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=65231&status=done&style=none&taskId=u713277f9-265d-46da-8bfe-d69c7ce78af&title=&width=665.6)<br />render函数用于渲染得到rgb图等，其所需参数与返回值如上所示，它调用了render_rays()函数对每个射线进行渲染<br />render_rays函数：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697123224479-2b40109e-4c44-46e8-95c3-1638f0e79acd.png#averageHue=%231a252f&clientId=u43ce8f52-26fb-4&from=paste&height=719&id=u88dc2ffd&originHeight=899&originWidth=800&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=93866&status=done&style=none&taskId=u2c38fe76-0ec8-4a92-a596-fb8a301c8d8&title=&width=640)<br />在render_rays函数中的`raw = network_query_fn(pts, viewdirs, network_fn)`实现了输出每个点的sigmma+rgb，得到raw，而后通过raw2outputs来完成积分过程进一步得到render函数的输出结果<br />raw2outputs函数：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2556769/1697123391633-b7e8813f-0ebf-417a-880c-57f3c9258923.png#averageHue=%231c2832&clientId=u43ce8f52-26fb-4&from=paste&height=218&id=u08331559&originHeight=272&originWidth=792&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=33242&status=done&style=none&taskId=u981eef35-fee0-4f37-b6b0-6642a1fc23f&title=&width=633.6)

---

9.后续代码内容：定期记载、生成观测视频、测试结果文件夹等
