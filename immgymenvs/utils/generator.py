from tasks.domain import Domain, gen_keypoints, compute_projected_points
from isaacgym.torch_utils import quat_from_euler_xyz, normalize
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Uniform
from algos.utils import *
from typing import Tuple
import numpy as np
import torch


def generatorBuilder(name, **kwargs):
    if name=="Card":
        return cardGenerator(**kwargs)
    elif name=="HiddenCard":
        return hiddenCardGenerator(**kwargs)
    elif name=="Flip":
        return flipGenerator(**kwargs) 
    elif name=="Hole":
        return holeGenerator(**kwargs)
    elif name=="Hole_wide":
        return holeWideGenerator(**kwargs)
    elif name=="Reorientation":
        return reoriGenerator(**kwargs)
    elif name=="Bookshelf":
        return bookGenerator(**kwargs)
    elif name=="Bump":
        return bumpGenerator(**kwargs)
    else:
        raise Exception(f"no available genrator for {name}")



class sampleGenerator:

    def __init__(self, model_pre, writer, map, IK_query_size, device) -> None:
        """
            model_pre: pi_pre model
            writer: tensorboard writer 
            map: mapping object
        """
        self.model_pre=model_pre
        self.writer=writer
        self.map=map
        self.IK_query_size=IK_query_size
        self.device=device

    def set_env(self, env: Domain):
        self.env = env

    def generate(self, size: int, frames: int, checkFeasibility: bool = False):
        """
            size: (int) total amount of contact samples.
            checkFeasibility: (boolean) If true check feasibility for contacts and guarantee (size) feasabile contacts.
        """
        buffer = {}
        keys = ['T_O', 'T_G', 'q_R', 'action', 'neglogp', 'mu', 'logstd', 'value']
        feasible = {}
        for key in keys:
            feasible[key] = []
        buffer['feasible'] = feasible
        if checkFeasibility:
            infeasible = {}
            for key in keys:
                infeasible[key] = []
            buffer['infeasible'] = infeasible
            nFeasible = 0 # The number of feasible samples.
            nInfeasible = 0 # The number of infeasible samples.
            while nFeasible <= size:
                T_O, T_G = self.sample(self.IK_query_size)
                q_R, feasibility, action, neglogp, mu, logstd, value = self.run_pre_contact_policy(T_O, T_G)
                isFeasible = (feasibility == 1)
                nFeasible += torch.count_nonzero(isFeasible)
                nInfeasible += torch.count_nonzero(torch.logical_not(isFeasible))
                for key in keys:
                    buffer['feasible'][key].append(eval(key)[isFeasible].clone())
                for key in keys:
                    buffer['infeasible'][key].append(eval(key)[torch.logical_not(isFeasible)].clone())
            for key, v in buffer['feasible'].items():
                buffer['feasible'][key] = torch.cat(v, dim=0)
            for key, v in buffer['infeasible'].items():
                buffer['infeasible'][key] = torch.cat(v, dim=0)
            infeasible_rate = nInfeasible / (nInfeasible + nFeasible)
            print(f"infeasible rate: {infeasible_rate}, feasible samples: {nFeasible}, infeasible samples: {nInfeasible}")
            if self.writer != None:
                self.writer.add_scalar('pre/rejection rate', infeasible_rate, frames)
        else:
            T_O, T_G = self.sample(size)
            q_R, _, action, neglogp, mu, logstd, value = self.run_pre_contact_policy(T_O, T_G)
            for key in keys:
                buffer['feasible'][key] = eval(key)
        return buffer

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Need task specific implementaion")

    def run_pre_contact_policy(self, T_O: torch.Tensor, T_G: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        action, value, neglogp, mu, logstd, _ = self.model_pre.step(torch.cat([T_O, T_G], -1))
        action_to_convert = action.clone().detach()
        action_to_convert[:, :4] = normalize(action_to_convert[:, :4])
        mapped = self.map.convert(action_to_convert, T_O)
        q_R = mapped[:, :-1]
        feasibility = mapped[:, -1]
        return q_R, feasibility, action, neglogp, mu, logstd, value

class cardGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        self.table_x=geometry["table"]["x"]
        self.table_y=geometry["table"]["y"]
        self.table_z=geometry["table"]["z"]

        cardDims=geometry["object"]
        cardDims=[cardDims["width"], cardDims["length"], cardDims["height"]]
        cardLength=np.sqrt(cardDims[0]**2+cardDims[1]**2)/2
        self.card_height=cardDims[2]
        table_dims=geometry["table"]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]]

        self.xsampler = Uniform((cardLength +xmin), (xmax - cardLength))
        self.ysampler = Uniform((cardLength +ymin), (ymax - cardLength))

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_O = self.xsampler.sample((size, 1)).to(self.device)
        y_O = self.ysampler.sample((size, 1)).to(self.device)
        x_G = torch.rand((size, 1), dtype=torch.float, device=self.device)*self.table_dims[0] - self.table_dims[0] / 2 + self.table_x 
        y_G = torch.rand((size, 1), dtype=torch.float, device=self.device)*self.table_dims[1] - self.table_dims[1] / 2 + self.table_y
        z = (self.table_z + self.table_dims[2]/2. + self.card_height / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        yaw = 2 * np.pi * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x_O, y_O, z, quat[:size]], -1)
        T_G = torch.cat([x_G, y_G, z, quat[size:]], -1)
        return T_O, T_G

class flipGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        zmin=geometry["zmin"]
        zmax=geometry["zmax"]

        cardDims=geometry["object"]
        cardDims=[cardDims["width"], cardDims["length"], cardDims["height"]]
        cardLength=np.sqrt(cardDims[0]**2+cardDims[1]**2)/2
        self.card_height=cardDims[2]
        self.cardDims=cardDims
        table_dims=geometry["table"]["base"]
        #self.chamfer=geometry["table"]["champer"]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]]
        self.y_surf=0.2
        #self.xsampler = Uniform((cardLength + xmin), (xmax - cardLength))
        self.xsampler = Uniform((-0.05 + 0.5), (0.5 + 0.05))
        # self.ysampler = Uniform((cardLength + ymin), (ymax - cardLength- geometry["chamfer"]))

        self.zsampler= Uniform((cardDims[1]/2 + self.table_dims[2]), (cardDims[1]/2 + self.table_dims[2])*1.1)

    def run_pre_contact_policy(
        self, T_O: torch.Tensor, T_G: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action, value, neglogp, mu, logstd, _ = self.model_pre.step(torch.cat([T_O, T_G], -1))
        action_to_convert = action.clone().detach()
        bias=torch.stack([torch.tensor([ 0.        ,  0.        , 0.38268343,  0.92387953], device=self.device)]*action.shape[0], 0)
        action_to_convert[:, :4] = normalize(action_to_convert[:, :4])
        mapped = self.map.convert(action_to_convert, T_O)
        q_R = mapped[:, :-1]
        feasibility = mapped[:, -1]
        return q_R, feasibility, action, neglogp, mu, logstd, value

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x = self.xsampler.sample((size*2, 1)).to(self.device)
        # y_O = self.ysampler.sample((size, 1)).to(self.device)
        x_O =0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)
        y_O=(0.2-0.014142-self.cardDims[1])*torch.ones((size, 1), dtype=torch.float, device=self.device)
        z_O = (0.4+0.5*self.cardDims[2]) * torch.ones((size, 1), dtype=torch.float, device=self.device) 

        #y_O =(0.2-0.035*0.5-0.0025*np.sqrt(3)/2)*torch.ones((size, 1), dtype=torch.float, device=self.device)
        #z_O = (0.4+0.035*np.sqrt(3)/2+0.0025*0.5) * torch.ones((size, 1), dtype=torch.float, device=self.device) 
        
        ## from t=250
        # x_O =0.5017*torch.ones((size, 1), dtype=torch.float, device=self.device)
        # y_O =0.1881*torch.ones((size, 1), dtype=torch.float, device=self.device)
        # z_O =  0.4346 * torch.ones((size, 1), dtype=torch.float, device=self.device)

        ##from t=0
        # x_O =0.50*torch.ones((size, 1), dtype=torch.float, device=self.device)
        # y_O =0.1159*torch.ones((size, 1), dtype=torch.float, device=self.device)
        # z_O =  0.4025 * torch.ones((size, 1), dtype=torch.float, device=self.device)
        #x_G =0.5*torch.ones((size, 1), dtype=torch.float, device=self.device)
        x_G = self.xsampler.sample((size, 1)).to(self.device)
        y_G= (-self.card_height / 2+self.y_surf) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        z_G = self.zsampler.sample((size, 1)).to(self.device)
        #z_G=self.zsample*torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll = np.pi/3*torch.zeros((size), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size), dtype=torch.float, device=self.device)
        yaw =  np.pi/3 * torch.rand((size), dtype=torch.float, device=self.device)+-np.pi/6
        quat_O = quat_from_euler_xyz(roll, pitch, yaw)

        ##t=250
        # quat_O=torch.zeros((size, 4), dtype=torch.float, device=self.device)
        # quat_O[:,0]=-0.7867
        # quat_O[:,1]=-0.0183
        # quat_O[:,2]=0.0144
        # quat_O[:,3]= 0.6169


        roll = -np.pi/2*torch.ones((size), dtype=torch.float, device=self.device)
        pitch = 2 * np.pi * torch.zeros((size), dtype=torch.float, device=self.device)
        yaw = torch.zeros((size), dtype=torch.float, device=self.device)
        quat_G = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x_O, y_O, z_O, quat_O], -1)
        T_G = torch.cat([x_G, y_G, z_G, quat_G], -1)
        return T_O, T_G

class holeGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        
        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        objDims=geometry["object"]
        self.objDims=[objDims["width"], objDims["length"], objDims["height"]]
        objLength=np.sqrt(self.objDims[0]**2+self.objDims[1]**2)/2
        self.obj_height=self.objDims[2]
        self.holeLength=geometry["holelength"]
        self.boxes=geometry["boxes"]

        # self.xsampler = Uniform((objLength + xmin), (xmax - objLength))
        self.ysampler = Uniform((-self.holeLength + self.objDims[1]/2), (self.holeLength -self.objDims[1]/2))

        # self.Gxsampler = Uniform((-self.holeLength/2+0.5+ self.objDims[0]/2), (self.holeLength/2+0.5-self.objDims[0]/2))
        # self.Gysampler = Uniform((-self.holeLength/2+ self.objDims[1]/2), (self.holeLength/2-self.objDims[1]/2))
        self.Gzsampler = Uniform((self.boxes["box5"]["height"]+ self.obj_height/2+0.005), (self.boxes["box5"]["height"]+self.boxes["box1"]["height"]))

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_O = self.xsampler.sample((size, 1)).to(self.device)
        y_O = self.ysampler.sample((size, 1)).to(self.device)
        x_O=(0.5+self.holeLength/2-self.objDims[0]/2)*torch.ones((size,1),dtype=torch.float, device=self.device)
        z_O = (self.boxes["box5"]["height"]+self.objDims[2]/2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        
        x_G = x_O.clone()
        y_G = torch.zeros((size,1),dtype=torch.float,device=self.device)
        x_G[:]=0.7-(0.2-self.holeLength/2)/2
        # x_G[cases==2]=(-self.holeLength/2+0.5+self.objDims[0]/2)
        # y_rand=torch.logical_or(cases==0,cases==2)
        # y_G[y_rand]=self.Gysampler.sample((y_rand.count_nonzero(),)).to(self.device)
        # y_G[cases==1]=(self.holeLength/2-self.objDims[1]/2)
        # y_G[cases==3]=(-self.holeLength/2+self.objDims[1]/2)
        # x_rand=torch.logical_or(cases==1,cases==3)
        # x_G[x_rand]=self.Gxsampler.sample((x_rand.count_nonzero(),)).to(self.device)
        z_G = (0.4+self.objDims[1]/2)*torch.ones((size,1 ), dtype=torch.float, device=self.device)
        roll = torch.zeros((size ), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size ), dtype=torch.float, device=self.device)
        yaw = torch.zeros((size ), dtype=torch.float, device=self.device)
        quat_O = quat_from_euler_xyz(roll, pitch, yaw)
        pitch[:]=np.pi/2
        quat_G = quat_from_euler_xyz(roll, pitch, yaw)
        
        T_O = torch.cat([x_O, y_O, z_O, quat_O], -1)
        T_G = torch.cat([x_G, y_G, z_G, quat_G], -1)
        return T_O, T_G

class holeWideGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        
        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        objDims=geometry["object"]
        self.objDims=[objDims["width"], objDims["length"], objDims["height"]]
        objLength=np.sqrt(self.objDims[0]**2+self.objDims[1]**2)/2
        self.obj_height=self.objDims[2]
        self.holeLength=geometry["holelength"]
        self.boxes=geometry["boxes"]

        # self.xsampler = Uniform((objLength + xmin), (xmax - objLength))
        # self.ysampler = Uniform((objLength + ymin), (ymax - objLength))

        # self.Gxsampler = Uniform((-self.holeLength/2+0.5+ self.objDims[0]/2), (self.holeLength/2+0.5-self.objDims[0]/2))
        # self.Gysampler = Uniform((-self.holeLength/2+ self.objDims[1]/2), (self.holeLength/2-self.objDims[1]/2))
        # self.Gzsampler = Uniform((self.boxes["box5"]["height"]+ self.obj_height/2+0.005), (self.boxes["box5"]["height"]+self.boxes["box1"]["height"]))
        self.xsampler = Uniform(0.38, 0.62)
        # self.ysampler = Uniform(0.08, 0.086)

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_O = self.xsampler.sample((size, 1)).to(self.device)
        # y_O = self.ysampler.sample((size, 1)).to(self.device)

        # cases=torch.randint(0,4,(size,1))
        cases = 0

        # x_O = self.xsampler.sample((size, 1)).to(self.device)
        # y_O = self.ysampler.sample((size, 1)).to(self.device)
        x_O = torch.full((size, 1), fill_value=0.5, dtype=torch.float, device=self.device)
        y_O = torch.full((size, 1), fill_value=-0.015, dtype=torch.float, device=self.device)
        z_O = (self.boxes["box1"]["height"]+self.objDims[2]/2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        
        # x_G = 0.5 * torch.ones((size,1),dtype=torch.float, device=self.device)
        x_G = self.xsampler.sample((size, 1)).to(self.device)
        y_G = 0.05 * torch.ones((size,1),dtype=torch.float, device=self.device)
        z_G = (0.4+self.objDims[1]/2) * torch.ones((size,1),dtype=torch.float, device=self.device)
        roll = torch.zeros((size,1), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size,1), dtype=torch.float, device=self.device)
        yaw = torch.zeros((size,1), dtype=torch.float, device=self.device)
       
        quat_O = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)
        roll[cases==0] = -np.pi/2
       
        quat_G = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)
        
        T_O = torch.cat([x_O, y_O, z_O, quat_O], -1)
        T_G = torch.cat([x_G, y_G, z_G, quat_G], -1)
        return T_O, T_G

class bookGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])

        geometry=kwargs.pop("geometry")
        self.size_x=geometry["bookshelf"]["book_size_x"]
        self.size_y=geometry["bookshelf"]["book_size_y"]
        self.size_z=geometry["bookshelf"]["book_size_z"]

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_O = (0.4+self.size_x/2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        y_O = torch.zeros((size, 1), dtype=torch.float, device=self.device)
        z_O = (0.4+self.size_z/2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll_O = torch.zeros((size), dtype=torch.float, device=self.device)
        pitch_O = torch.zeros((size), dtype=torch.float, device=self.device)
        yaw_O = torch.zeros((size), dtype=torch.float, device=self.device)
        quat_O = quat_from_euler_xyz(roll_O, pitch_O, yaw_O)
        T_O = torch.cat([x_O, y_O, z_O, quat_O], -1)

        x_G = (0.4+self.size_x/2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        y_G = (0.0)*torch.ones((size, 1), dtype=torch.float, device=self.device)
        z_G = (0.4+self.size_z/2+0.005) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll_G = torch.zeros((size), dtype=torch.float, device=self.device)
        pitch_G = np.pi * (-1.0/6.0) * torch.ones((size), dtype=torch.float, device=self.device)
        yaw_G = torch.zeros((size), dtype=torch.float, device=self.device)
        quat_G = quat_from_euler_xyz(roll_G, pitch_G, yaw_G)
        T_G = torch.cat([x_G, y_G, z_G, quat_G], -1)
        return T_O, T_G
        
class reoriGenerator(sampleGenerator):
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        
        
        self.obj_dims=kwargs.pop("objHeight")
        objeLength=np.sqrt(self.obj_dims[0]**2+self.obj_dims[1]**2)/2
        self.table_dims=kwargs.pop("tableDims")
        self.hole_dims=kwargs.pop("holeDims")

        self.xsampler = Uniform((objeLength - self.hole_dims[0]), (self.hole_dims[0] - objeLength))
        self.ysampler = Uniform((objeLength-self.hole_dims[1]), (self.hole_dims[1] - objeLength))

        self.zsampler= Uniform(self.table_dims[2]-self.hole_dims[2]+self.obj_dims[2]/2, self.table_dims[2])

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

class bumpGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        geometry=kwargs.pop("geometry")
        xmin=geometry["xmin"]
        xmax=geometry["xmax"]

        ymin=geometry["ymin"]
        ymax=geometry["ymax"]

        bump=geometry["obstacle"]
        self.bumpDims=[bump["width"], bump["length"], bump["height"]]

        boxDims=geometry["object"]
        self.boxDims=[boxDims["width"], boxDims["length"], boxDims["height"]]
        self.boxLength=np.sqrt(self.boxDims[0]**2+self.boxDims[1]**2)/2
        table_dims=geometry["table"]
        self.table_dims=[table_dims["width"], table_dims["length"], table_dims["height"]]
        self.safety_margin = 0.01
        self.both_side = geometry["both_side"]
        if self.both_side:
            self.left_right_sampler = Bernoulli(0.5)

        self.xsampler = Uniform((self.boxLength + xmin), (xmax - self.boxLength))
        self.ysampler = Uniform((self.bumpDims[1] / 2 + self.boxLength), (self.table_dims[1] / 2 - self.boxLength))

        self.Gysampler = Uniform(
            low=(self.safety_margin - self.bumpDims[1] / 2),
            high=(self.table_dims[1] / 2 - 2 * self.boxLength - self.safety_margin)
        )

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.xsampler.sample((size*2, 1)).to(self.device)
        y_O = -self.ysampler.sample((size, 1)).to(self.device)
        z_O = (self.table_dims[2] + self.boxDims[2] / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)

        y_G = self.Gysampler.sample((size, 1)).to(self.device)
        onbump = (y_G <= (self.bumpDims[1] / 2 - self.safety_margin))
        z_G = torch.zeros_like(z_O)
        z_G[onbump] = self.table_dims[2] + self.bumpDims[2] + self.boxDims[2] / 2
        z_G[~onbump] = self.table_dims[2] + self.boxDims[2] / 2
        y_G[~onbump] += (self.safety_margin + self.boxLength)
        if self.both_side:
            left_right = self.left_right_sampler.sample((size,))
            right = (left_right > 0.5)
            y_O[right] *= -1
            left_right = self.left_right_sampler.sample((size,))
            left = (left_right < 0.5)
            y_G[left] *= -1

        roll = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        pitch = (np.pi / 2) * torch.randint(0, 4, (size * 2,), device=self.device)
        yaw = (2 * np.pi) * torch.rand((size * 2), dtype=torch.float, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x[:size], y_O, z_O, quat[:size]], -1)
        T_G = torch.cat([x[size:], y_G, z_G, quat[size:]], -1)
        return T_O, T_G

class hiddenCardGenerator(sampleGenerator):

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["model_pre"], kwargs["writer"], kwargs["map"], kwargs["IK_query_size"], kwargs["device"])
        geometry = kwargs.pop("geometry")
        xmin = geometry["xmin"]
        xmax = geometry["xmax"]

        ymin = geometry["ymin"]
        ymax = geometry["ymax"]

        self.table_x = geometry["tableX"] if "tableX" in geometry else 0.5
        self.table_y = geometry["tableY"] if "tableY" in geometry else 0.0

        cardDims = geometry["object"]
        cardDims = [cardDims["width"], cardDims["length"], cardDims["height"]]
        cardLength = np.sqrt(cardDims[0]**2 + cardDims[1]**2) / 2
        self.card_height = cardDims[2]
        table_dims = geometry["table"]
        self.table_dims = [table_dims["width"], table_dims["length"], table_dims["height"]]

        self.xsampler = Uniform(xmin, (xmax - cardLength))
        self.ysampler = Uniform(ymin, ymax)

    def sample(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_O = torch.ones((size, 1), dtype=torch.float, device=self.device) * 0.55
        y_O = torch.zeros((size, 1), dtype=torch.float, device=self.device)
        x_G = self.xsampler.sample((size, 1)).to(self.device)
        y_G = self.ysampler.sample((size, 1)).to(self.device)
        z = (self.table_dims[2] + self.card_height / 2) * torch.ones((size, 1), dtype=torch.float, device=self.device)
        roll = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        pitch = torch.zeros((size * 2), dtype=torch.float, device=self.device)
        yaw = 2 * np.pi * torch.rand((size * 2), dtype=torch.float, device=self.device)
        # yaw[:size] = 0
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        T_O = torch.cat([x_O, y_O, z, quat[:size]], -1)
        T_G = torch.cat([x_G, y_G, z, quat[size:]], -1)
        return T_O, T_G
