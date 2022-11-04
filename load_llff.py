import numpy as np
import torch
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius) # 先做平移
    c2w = rot_phi(phi/180.*np.pi) @ c2w # 绕x轴旋转,@是矩阵的叉乘
    c2w = rot_theta(theta/180.*np.pi) @ c2w # 绕y轴再旋转
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    '''
       .npy文件是一个shape为（20，17），dtype为float64的array，20代表数据集的个数（一共有20张图片），17代表位姿参数。
       poses_arr[:, :-2]代表取前15列，为一个（20,15）的array，
       reshape([-1, 3, 5])代表将（20,15）的array转换为（20,3,5）的array，也就是把15列的一维数据变为3*5的二维数据。
       transpose([1,2,0])则是将array的坐标系调换顺序，0换到2, 1、2换到0、1，shape变为（3,5,20）;
       最后poses输出的是一个（3,5,20）的array
    '''
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy')) # 17会转换为3x5的矩阵和两个深度值，视角到场景的最近和最远距离
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 每张图片的外参位姿
    bds = poses_arr[:, -2:].transpose([1,0]) # 近远距离，即深度范围

    # img0是20张图像中的第一张图像的路径名称，'./data/nerf_llff_data/fern\\images\\IMG_4026.JPG'
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''

    # 判断是否有下采样的相关参数，如果有，则对图像进行下采样
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor]) # 将所有的图片进行尺寸的缩小
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    # 判断是否存在下采样的路径'./data/nerf_llff_data/fern\\images_8'
    imgdir = os.path.join(basedir, 'images' + sfx) # 保存下采样8倍之后的存放路径
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    # 判断pose数量与图像个数是否一致，
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    #  获取处理后的图像shape，sh=（378,504,3）=（3024/8, 4032/8, 3）
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # factor下采样之后内参的高宽需要改变
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # 注意焦距也要跟着一起变
    '''
    sh[:2]存的是前两个数据，也就是图片单通道的大小（378,504）；
    np.array(sh[:2]).reshape([2, 1])将其先array化后reshape为2*1的大小：array([[378],[504]])
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])则表示将poses中3*5矩阵的前两行的第5列存放height=378，width=504；
    poses[2, 4, :]则表示第三行第5列的存放图像的分辨率f，更新f的值最后为3261/8=407.56579161
    另外，3*5矩阵的前3行3列为旋转变换矩阵，第4列为平移变换矩阵，第5列为h、w、f；
    '''
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    # 读取所有图像数据并把值缩小到0-1之间，imgs存储所有图片信息，大小为（378,504,3,20）
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles] # 把图像的channel维度归一化
    imgs = np.stack(imgs, -1)   # 把列表全部展开
    
    print('Loaded image data', imgs.shape, poses[:,-1,0]) # 打印处理后的图片的信息，并且打印相机的内参
    return poses, bds, imgs # 返回相机的参数矩阵，返回相机的近远距离，返回下采样和归一化后的图片数据

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    '''
    np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)指的是进行矩阵变换，将poses每个通道的第0行的相反数和第1行互换位置；
    紧接着用np.moveaxis(poses, -1, 0).astype(np.float32)将坐标轴的第-1轴换到第0轴；
    得到的poses的shape为（20,3,5）
    imgs同理，变换完的shape为（20,378,504,3）
    bds的shape为（20,2)
    '''
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    # 深度边界和平移变换向量一同进行缩放
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc # SC是进行边界缩放的比例
    bds *= sc  # 缩放到1，6的深度左右？
    
    if recenter: # 位姿中心化
        # 修改pose(图像数，3（x,y,z），5) 前四列的值，只有最后一列 （高，宽，焦距）不变
        # 计算poses的均值，将所有pose做该均值的逆转换，即重新定义了世界坐标系，原点大致在被测物中心；
        poses = recenter_poses(poses)

    if spherify: # 不走这一步
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # shape=(3,5)相当于平均了20张图像的外参？
        c2w = poses_avg(poses)
        # 经过recenter pose，求均值逆变换处理后，旋转矩阵变为单位阵，平移矩阵变为0
        '''[[ 1.0000000e+00  0.0000000e+00  0.0000000e+00  1.4901161e-09]
        [ 0.0000000e+00  1.0000000e+00 -1.8730975e-09 -9.6857544e-09]
        [-0.0000000e+00  1.8730975e-09  1.0000000e+00  0.0000000e+00]]
        '''
        print('recentered', c2w.shape)
        print(c2w[:3,:4]) # 打印外参矩阵:相机坐标系->世界坐标系 ,第5列是内参参数

        ## Get spiral
        # Get average pose,不知道这个代表什么？
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz # 这个focal不是相机的焦距

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2 # 指的是光线中的一小格？？？
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0) # 弧度制？
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # 渲染时期的位姿
        # 一个list，有120（由N_views决定）个元素，每个元素shape（3，5），内参没有区别，但是外参有区别
        # render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],0)  # -π到π转一圈生成旋转视频

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1) # 每张图片的外参与考虑所有图片的外参的平移矩阵的差值
    i_test = np.argmin(dists) # 把离的最近的那个作为测试样本？
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



