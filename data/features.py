# -^- coding:utf-8 -^-
"""
The implementation of feature computation in an object oriented manner

code paraphased from feature_computation.py and feature_compuatation_1d.py
"""

import autograd.numpy as np
from autograd import grad, jacobian

DT = 0.1
VLIM = 11.7
WHOLEPROCESSNORM = True
SINGULARVLAUENORM = False
L2NotSquared = True # from the request of Zhengwu at 2020-01-10, is this is true, the _L2 means sqrt(L2...)

def getSetting():
    """
        return a string about all the flags in this file
    """
    return "### featuresSetting:\n\n" + "\n".join(["%s : %s"%(k,str(v)) for k,v in 
       [("DT",DT),
        ("**VLIM**",VLIM),
        ("WHOLEPROCESSNORM",WHOLEPROCESSNORM),
        ("SINGULARVLAUENORM",SINGULARVLAUENORM),
        ("**L2NotSquared**",L2NotSquared)]])


def distance(a,b):
    """
    calculate the euclid distance between two points
    """
    return np.sum((a - b)**2)**0.5

def curveMatch(crv1,crv2):
    """
    input two curves(each is a spline) and match the crv2 to the crv1. 
        i.e. to find a closest point in crv1 for every point in crv2
    return the a list of index of the match, same length with crv2
    """
    i = 0
    lencrv1 = len(crv1)
    res = [0]*len(crv2)
    for j, p in enumerate(crv2):
        i = res[max(0,j-1)]
        min_dis = distance(crv1[i],p)
        min_ind = i
        while(i<lencrv1-1):
            d = distance(crv1[i+1],p)
            if(d < 2* min_dis):
                i += 1
                if( d < min_dis):
                    min_ind = i
                    min_dis = d
            else:
                break
        
        res[j] = min_ind
    return res

def footP(p1,p2,p0):
    """
    calculate the position of the foot of the point p0 on the line formed by p1 and p2
    """
    p12 = p2-p1
    len12 = np.sum(p12**2)
    x = (p12[0]**2*p0[0]+p1[0]*p12[1]*(p2[1] - p0[1]) + p2[0]*p12[1]*(p0[1]-p1[1]))/len12
    y = (p12[1]**2*p0[1]+ p1[1]*p12[0]*(p2[0] - p0[0]) + p2[1]*p12[0]*(p0[0]-p1[0]))/len12
    return np.array((x,y))



def closestPoint(curv1,curv2):
    """
    find each point in curv2 a cloest point on the spline curv1
    each curv is represented as an (n,2) array
    also returns the index list, each element representing the index on the spline that the projection belongs to 
    """
    matchp = curveMatch(curv1,curv2)
    refinds = matchp[:]
    res = np.zeros_like(curv2)
    for i,p in enumerate(curv2):
        mchi = matchp[i]
        mchp = curv1[mchi]
        resp = curv1[mchi] # result point
        p21 = mchp - p
        # check two foots of the segment beside the matched point
        if(mchi==0):
            resp = footP(curv1[mchi+1],mchp,p) # elongating the first point
        elif(mchi>0 and  p21.dot( curv1[mchi-1] - mchp )<0):
            resp = footP(curv1[mchi-1],mchp,p) # the projection is on the last point
            refinds[i] = mchi - 1
        elif(mchi<len(curv1)-1 and  p21.dot( curv1[mchi+1] - mchp )<0):
            resp = footP(curv1[mchi+1],mchp,p)
        elif(mchi==len(curv1)-1):
            resp = footP(curv1[mchi-1],mchp,p) # elongating the last point
        res[i]= resp 
    return res, refinds


class LazyFunc:
    """
    Function factories to only generate the function when it is called
        This is used to save the necessary computation such as ref_d and ref_s
    """
    def __init__(self,factry):
        self.factry = factry # the expansive factory method to generate the function
        
    def __call__(self,*args):
        # print("called Lazy")
        try:
            return self.exe(*args)
        except AttributeError:
            self.exe = self.factry()
        return self.exe(*args)
        

class Features:
    def __init__(self, vec, referenceCurv = None, v_des_func = None,**kwargs):
        """
        vec is the vector with half of it x and half of it y
        
        dt is the dt between two time points in the list

        v_lim is the desired speed of the car, Now the default value is the m/s for 25 mile/hour

        referenceCurv is the reference curve, [n,i] n is the nth point, i is eigher x or y

        the private functions returns functions 

        v_des_func: a function that outputs desired speed given location
        """
        dt=DT
        v_lim = VLIM
        self.vec = vec
    
        self.vectors = {} # vectors[n] is the information of the n'th derivative, for example pos, velocity, acceleration, jerk
        
        self.vec_len = int(vec.shape[0] / 2)
        self.dt = dt
        # self.inputVector = np.concatenate([self._x(2),self._y(2)]) # the action space is the acceleration of the car
        self._x = lambda vec: vec[:self.vec_len]
        self._y = lambda vec: vec[self.vec_len:]
        self._vx = self._diffdt(self._x)
        self._vy = self._diffdt(self._y)
        self._theta = lambda vec: np.arctan2(self._vx(vec),self._vy(vec))
        self._v =  self._distance(self._vx,self._vy)
        self._ax = self._diffdt(self._vx)
        self._ay = self._diffdt(self._vy)
        self._ds = self._distance(self._diff(self._x),self._diff(self._y))
        self._a = self._distance(self._ax,self._ay)

        self._s = self._cumsum(self._ds)
        
        self._alon = self._normalize(self._aPlon(self._x,self._y), self._avrun(self._v)) # (a_x*v_x + a_y*v_y) / v
        self._alat = self._normalize(self._crossMul(self._x,self._y), self._avrun(self._v)) # (a_x*v_x + a_y*v_y) / v

        self._jlon = self._normalize(self._jPlon(self._x,self._y), self._avrun(self._avrun(self._v))) # (a_x*v_x + a_y*v_y) / v
        # smooth J_lon
        # self._jlon = self._normalize(self._jPlon(self._avrun(self._avrun(self._x)),self._avrun(self._avrun(self._y))), self._avrun(self._avrun(self._avrun(self._avrun(self._v))))) # (a_x*v_x + a_y*v_y) / v
        self._jlat = self._normalize(self._crossMul(self._vx,self._vy) , self._avrun(self._a)) # (a_x*v_x + a_y*v_y) / v
        self._kappa = self._kappa_(self._x,self._y)

        self.referenceCurv = referenceCurv # the raw points of the reference Curv

        # self._ref_ds = self._ref_ds_()
        self._ref_ds = LazyFunc(self._ref_ds_)
        # self._ref_d = self._ref_d_() # the deviation with the reference curve
        self._ref_d = LazyFunc(self._ref_d_)
        self._ref_s = self._cumsum(self._ref_ds)
        self.v_lim = v_lim

        self._final_v = lambda vec: self._v(vec)[-1] # the finale speed

        self._ref_sinphi =  self._normalize(self._ref_ds,self._ds) # the sin of angel formed between the car trajectory and the ref trajectory
        self.features ={"L2_a_lon":self._L2(self._alon,self._const(0)),
                        "L1_a_lon":self._L1(self._alon,self._const(0)),
                        "Linf_a_lon":self._Linf(self._alon,self._const(0)),

                        "L2_a_lat":self._L2(self._alat,self._const(0)),
                        "L1_a_lat":self._L1(self._alat,self._const(0)), 
                        "Linf_a_lat":self._Linf(self._alat,self._const(0)), 

                        "L2_j_lon":self._L2(self._jlon,self._const(0)),
                        "L1_j_lon":self._L1(self._jlon,self._const(0)),
                        "Linf_j_lon":self._Linf(self._jlon,self._const(0)),

                        "L2_j_lat":self._L2(self._jlat,self._const(0)),
                        "L1_j_lat":self._L1(self._jlat,self._const(0)), 
                        "Linf_j_lat":self._Linf(self._jlat,self._const(0)), 
                        
                        # Note: `v_des` and `abs_v_des` are identical, they are used interchangablly for historical reason
                        "L2_v_des":self._L2(self._v,self._const(self.v_lim)),
                        "L1_v_des":self._L1(self._v,self._const(self.v_lim)),
                        "Linf_v_des":self._Linf(self._v,self._const(self.v_lim)),

                        "L2_abs_v_des":self._L2(self._abs(self._add(self._neg(self._v),self._const(self.v_lim))),self._const(0)),
                        "L1_abs_v_des":self._L1(self._abs(self._add(self._neg(self._v),self._const(self.v_lim))),self._const(0)),
                        "Linf_abs_v_des":self._Linf(self._abs(self._add(self._neg(self._v),self._const(self.v_lim))),self._const(0)),

                        "L2_ref_d":self._L2(self._ref_d,self._const(0)),
                        "L1_ref_d":self._L1(self._ref_d,self._const(0)),
                        "Linf_ref_d":self._Linf(self._ref_d,self._const(0)),

                        "L2_ref_a_d":self._L2(self._diffdt(self._ref_d),self._const(0)),
                        "L1_ref_a_d":self._L1(self._diffdt(self._ref_d),self._const(0)),
                        "Linf_ref_a_d":self._Linf(self._diffdt(self._ref_d),self._const(0)),

                        "L2_ref_a_s":self._L2(self._diff(self._ref_ds),self._const(0)),
                        "L1_ref_a_s":self._L1(self._diff(self._ref_ds),self._const(0)),
                        "Linf_ref_a_s":self._Linf(self._diff(self._ref_ds),self._const(0)),

                        "L2_ref_sinphi":self._L2(self._ref_sinphi,self._const(0)),
                        "L1_ref_sinphi":self._L1(self._ref_sinphi,self._const(0)),
                        "Linf_ref_sinphi":self._Linf(self._ref_sinphi,self._const(0)),

                        "L2_final_v": self._L2(self._final_v,self._const(0)),
                        "L1_final_v": self._L1(self._final_v,self._const(0)),
                        "Linf_final_v": self._Linf(self._final_v,self._const(0))
                        }

        if(v_des_func is not None):
            self.features["L2_v_des_func"] = self._v_des_delta_(v_des_func,self._L2)
            self.features["L1_v_des_func"] = self._v_des_delta_(v_des_func,self._L1)
            self.features["Linf_v_des_func"] = self._v_des_delta_(v_des_func,self._Linf)

    @property
    def refcurv(self):
        try:
            return self.refcurvValue
        except AttributeError:
            self.refcurvValue, self.refindsValue = self._refcurv(self.vec)
            return self.refcurvValue

    @property
    def refinds(self):
        try:
            return self.refindsValue
        except AttributeError:
            self.refcurvValue, self.refindsValue = self._refcurv(self.vec)
            return self.refindsValue

    @property 
    def refInfo_rdxrdy(self):
        try:
            return self.refInfo_rdxrdyValue
        except AttributeError:
            rx = self.refcurv[:self.vec_len]
            ry = self.refcurv[self.vec_len:]
            rdx = np.concatenate([[0], rx[1:] - rx[:-1]],axis = 0)
            rdy = np.concatenate([[0], ry[1:] - ry[:-1]],axis = 0)

            # h,t = 0,len(rdx)-1 # the index between which to fix zeros
            nonzeroinds = np.where((rdx!=0) + (rdy!=0))[0]
            h = np.min(nonzeroinds)
            rdx[:h]+=rdx[h]
            rdy[:h]+=rdy[h]
            t = np.max(nonzeroinds)
            rdx[t+1:]+=rdx[t]
            rdy[t+1:]+=rdy[t]
            zeroinds = np.where((rdx==0) * (rdy==0))[0]
            while(len(zeroinds)):
                rdx[zeroinds] = (rdx[zeroinds-1] + rdx[zeroinds+1])/2
                rdy[zeroinds] = (rdy[zeroinds-1] + rdy[zeroinds+1])/2
                zeroinds = np.where((rdx==0) * (rdy==0))[0]
            self.refInfo_rdxrdyValue  = (rdx,rdy)
            return self.refInfo_rdxrdy

    def _v_des_delta_(self,v_des_func,norm_func):
        x_array = self.vec[:self.vec_len]
        y_array = self.vec[self.vec_len:]
        v_des_array = v_des_func(x_array,y_array)[:-1] # here v_des_func returns the length same with x_array and y_array
        return norm_func(self._v ,self._const(v_des_array))

    def _distance(self,prepx,prepy):
        return self._pow( self._add(self._pow(prepx,2) , self._pow(prepy,2)),0.5) 

    def _diffdt(self, prep):
        def ret_func(vec):
            vec = prep(vec)
            return (vec[1:] - vec[:-1])/self.dt
        return ret_func
    
    def _diff(self, prep):
        def ret_func(vec):
            vec = prep(vec)
            return vec[1:] - vec[:-1]
        return ret_func
    
    def _cumsum(self,prep):
        def ret_func(vec):
            vec = prep(vec)
            vec = np.concatenate([np.array([0]),vec],axis = 0)
            return np.cumsum(vec)
        return ret_func

    def _avrun(self,prep):
        # this function do smoothing 
        # The main purpose is to shorten the list by 1. 
        #   for example, only in this way, can velocity vector be multiplyed with accleration vector
        def ret_func(vec):
            vec = prep(vec)
            return (vec[1:] + vec[:-1])/2
        return ret_func
    
    def _limit(self,prep1,lwlim=None,uplim=None):
        def ret_func(vec):
            vec = prep1(vec)
            # outlimit = vec > lim
            # vec[outlimit] = lim
            vec = vec.clip(lwlim,uplim)
            return vec
        return ret_func 

    def _min(self,prep1,prep2):
        return lambda vec: np.min(np.concatenate([ prep1(vec)[None,...],prep2(vec)[None,...]], axis = 0),axis = 0)

    def _sqrt(self,prep):
        return lambda vec: np.sqrt(np.abs(prep(vec))+1e-20)

    def _const(self,cons):
        return lambda vec: cons

    def _scale(self,x,prep):
        return lambda vec: x*prep(vec)

    def _product(self,prep0,prep): # hadamard prod
        return lambda vec: prep0(vec)*prep(vec)

    def _abs(self,prep):
        return lambda vec: np.abs(prep(vec))

    def _L1(self, prep1, prep2):
        if(WHOLEPROCESSNORM):
            return lambda vec: np.mean(abs(prep1(vec)-prep2(vec)))
        else:
            return lambda vec: np.sum(abs(prep1(vec)-prep2(vec)))
    
    def _L2(self, prep1, prep2):
        if(WHOLEPROCESSNORM):
            if(L2NotSquared):
                return lambda vec: np.sqrt(np.mean((prep1(vec)-prep2(vec))**2)+1e-20)
            else:
                return lambda vec: np.mean((prep1(vec)-prep2(vec))**2) + 1e-9
        else:
            if(L2NotSquared):
                return lambda vec:np.sqrt(np.sum((prep1(vec)-prep2(vec))**2)+1e-20)
            else:
                return lambda vec: np.sum((prep1(vec)-prep2(vec))**2) + 1e-9
    

    def _Linf(self, prep1, prep2):
        return lambda vec: np.max(abs(prep1(vec)-prep2(vec)))

    def _add(self, prep1, prep2):
        return lambda vec: prep1(vec) + prep2(vec)
    
    def _neg(self,prep):
        return lambda vec: -prep(vec)

    def _normalize(self,prep1,prep2):
        return lambda vec: prep1(vec) / (prep2(vec) + 1e-9)

    
    def _pow(self,prep,times):
        return lambda vec: prep(vec)**times + 1e-9
    
    def _e(self,prep,p = 1,d = 0):
        return lambda vec: np.exp(p*prep(vec)+d)

    # (v_x(vec) * acc_y(vec) - acc_x(vec) * v_y(vec)) / (v_x(vec) ** 2. + v_y(vec) ** 2.) ** 1.5
    def _crossMul(self,prepx,prepy):
        return lambda vec: (self._avrun(self._diffdt(prepx))(vec) * self._diffdt(self._diffdt(prepy))(vec)
                            - self._avrun(self._diffdt(prepy))(vec) * self._diffdt(self._diffdt(prepx))(vec) )
    def _aPlon(self,prepx,prepy):
        """
        a projection on longitude
        """
        return lambda vec: (self._avrun(self._diffdt(prepx))(vec) * self._diffdt(self._diffdt(prepx))(vec)
                            + self._avrun(self._diffdt(prepy))(vec) * self._diffdt(self._diffdt(prepy))(vec) )

    def _jPlon(self,prepx,prepy):
        """
        a projection on longitude
        """
        return lambda vec: (self._avrun(self._avrun(self._diffdt(prepx)))(vec) * self._diffdt(self._diffdt(self._diffdt(prepx)))(vec)
                            + self._avrun(self._avrun(self._diffdt(prepy)))(vec) * self._diffdt(self._diffdt(self._diffdt(prepy)))(vec))
                                                        
    def _kappa_(self,prepx,prepy):
        return lambda vec: (self._crossMul(prepx,prepy)(vec) / 
                        (self._avrun(self._diffdt(prepx))(vec) ** 2 + self._avrun(self._diffdt(prepy))(vec) ** 2 + 1e-9)**1.5 )
    
    def _refcurv(self,vec):
        """
        the input vec is a list contains [x,....x, y,.... y]
        """
        if(self.referenceCurv is None):
            return None
        # print("vecshape",vec.shape)
        # point_vec = np.concatenate((vec[:self.vec_len,None],vec[self.vec_len:,None]),axis = 1)
        point_vec = vec.reshape(2,-1).T
        # cv_ind = curveMatch(self.referenceCurv, point_vec)
        # print(point_vec.shape)
        refcurv, refinds = closestPoint(self.referenceCurv, point_vec)
        # refcurv = np.concatenate((refcurv[:,0],refcurv[:,1]), axis = 0)
        refcurv = refcurv.T.reshape(-1)
        return refcurv, refinds

    def _ref_d_(self):
        """
        return the function to calculate the latter deviation to ref line
        """
        if(self.referenceCurv is None):
            return None
        rdx,rdy = self.refInfo_rdxrdy
        # assert(not np.any((rdx==0)*(rdy==0)))
        def ret_func(vec):
            d = vec - self.refcurv
            dx = d[:self.vec_len]
            dy = d[self.vec_len:]
            # return (dx**2 + dy**2+1e-9)**0.5
            return (rdx * dy - rdy * dx)/np.sqrt(rdx*rdx + rdy*rdy)
        return ret_func
    
    def _ref_ds_(self):
        """
        return the distance of the vector projected onto the ref line
        """
        if(self.referenceCurv is None):
            return None
        # calculate the first dev of the ref path
        rdx,rdy = self.refInfo_rdxrdy
        rdx,rdy = rdx[1:],rdy[1:]
        rds = (rdx**2 + rdy**2+1e-9)**0.5 
        rds += 1e-9 # for the numerical consideration
        def ret_func(vec):    
            dx = self._diff(self._x)(vec)
            dy = self._diff(self._y)(vec)
            # calculate the projection 
            return (dx*rdx + dy*rdy) /rds
        return ret_func

    def featureGradJacobNormalizer(self,feature,singulervaluenorm = SINGULARVLAUENORM, computeHessian = True):
        featurefun = self.features[feature]
        # forward = featurefun(self.vec)
        # assert( not np.isnan(forward).any() )
        g = jacobian(featurefun)(self.vec)[:]
        # assert( not np.isnan(g).any() )
        if(computeHessian):
            H = jacobian(grad(featurefun))(self.vec)[:,:]
            assert( not np.isnan(H).any() )
        else:
            H = 1
        if(computeHessian and singulervaluenorm):
            normalizer = np.linalg.svd(H)[1][0]
        else:
            normalizer = 1

        g,H = g/normalizer, H/normalizer
    
        return g,H,normalizer


    def featureValue(self,feature):
        featurefun = self.features[feature]
        forward = featurefun(self.vec)
        return forward
    

    

class HighDFeatures(Features):
    """
    This class provides more features about the interacting cars
    """
    def __init__(self, vec, referenceCurv=None, v_des_func=None,
            pcd_sor = None,flw_sor = None,pcd_tar = None,flw_tar= None,**kwargs):
        """
        
        :param pcd_sor: (preceding_source_lane, car_width, car_height)
               flw_sor: (following_source_lane, car_width, car_height)
               pcd_tar: (preceding_target_lane, car_width, car_height)
               flw_tar: (following_target_lane, car_width, car_height)
        :return: a list containing all tracks as dictionaries.
        """        
        super().__init__(vec, referenceCurv=referenceCurv, v_des_func=v_des_func)
        dt=DT
        v_lim = VLIM
        self.pcd_sor, self.pcd_sor_w, self.pcd_sor_h = pcd_sor
        self.flw_sor, self.flw_sor_w, self.flw_sor_h = flw_sor
        self.pcd_tar, self.pcd_tar_w, self.pcd_tar_h = pcd_tar
        self.flw_tar, self.flw_tar_w, self.flw_tar_h = flw_tar

        self._IDM_astar = self._IDM_astar_()

        self.features["L1_safe"] = self._add(self._L1(self._safty(self.pcd_tar, self.pcd_tar_w, self.pcd_tar_h),self._const(0)),
                                        self._L1(self._safty(self.flw_tar, self.flw_tar_w, self.flw_tar_h),self._const(0)))
        self.features["L2_safe"] = self._add(self._L2(self._safty(self.pcd_tar, self.pcd_tar_w, self.pcd_tar_h),self._const(0)),
                                        self._L2(self._safty(self.flw_tar, self.flw_tar_w, self.flw_tar_h),self._const(0)))
        self.features["Linf_safe"] = self._add(self._Linf(self._safty(self.pcd_tar, self.pcd_tar_w, self.pcd_tar_h),self._const(0)),
                                        self._Linf(self._safty(self.flw_tar, self.flw_tar_w, self.flw_tar_h),self._const(0)))
        
        self.features["L1_a_IDM"] = self._L1(self._IDM_astar , self._alon)

        self.features["L2_a_IDM"] = self._L2(self._IDM_astar , self._alon)

        self.features["Linf_a_IDM"] = self._Linf(self._IDM_astar , self._alon)
        
    def _safty(self,target,targeta,targetb):
        """
        calculate the safty loss towards other cars e^{-\frac{(x-x_0)^2}{a^2}+\frac{(y-y_0)^2}{b^2}}:
        :param target: n*2 matrix representing the traj of other cars (the time is matched between trajs)
        :param targeta: 
        :param targetb: The long and short axis of the car
        :return:  a function lambda vec -> double(safety loss)
        
        TODO: this is typically too small when comparing to other loss, consider how to deal with it.
        """
        targetx = target[:,0]
        targety = target[:,1]
        return self._e(self._neg(self._distance(
                self._normalize(self._add(self._x,self._neg(self._const(targetx))),self._const(targeta)),
                self._normalize(self._add(self._y,self._neg(self._const(targety))),self._const(targetb)))))

    def _IDM_astar_(self, v0 = 0.5, s0 =2, s1 = 3, T = 1.6,a  =0.73, b = 1.67, gamma = 4):
        # following the paper: https://tigerprints.clemson.edu/cgi/viewcontent.cgi?article=2936&context=all_theses
        halflen = int(len(self.pcd_sor)/2)
        frontcar = np.concatenate((self.pcd_sor[:halflen],self.pcd_tar[halflen:]),axis=0)
        frontcarx = frontcar[:,0]
        frontcary = frontcar[:,1]
        distanceFunc = self._distance(
                self._normalize(self._add(self._x,self._neg(self._const(frontcarx))),self._const(self.pcd_sor_w)),
                self._normalize(self._add(self._y,self._neg(self._const(frontcary))),self._const(self.pcd_sor_h)))
        def returnFunc(vec):
            s = self._avrun(self._avrun(distanceFunc))(vec)
            v = self._avrun(self._v)(vec)
            dv = self._a(vec)
            sStar = s0 + s1*np.sqrt(v/v0) + T*v + v*dv/2*np.sqrt(a*b)
            aStar = 1 - (v/v0)**gamma - (sStar/s)**2
            return aStar
        return returnFunc


class DoubleCarSRFeatures(Features):
    """
    This class provides safety features about the double cars in the SR scenerio
    """
    def __init__(self, vec, other_vec, referenceCurv=None, v_des_func=None,
                    whole = None, otherwhole = None, actInd = (None,None), intersectInd = (None,None), intersectPTs = (None,None),otherDist = None,**kwargs):
        """
        :param other_vec        : The x_y vec of the interaction car # NOTE THIS IS A DEPARCAtED INTERFACE
        :param referenceCurv    : The x_y vec to calculate the ref_ features
        :param whole            : The whole trajectory of vec
        :param otherwhole       : The whole trajectory of other_vec
        :param actInd           : (indego, indother)The index on the whole trajectories of the vec
        :param intersectInd     : (indego, indother)The index on the whole trajectories of the intersection point
        :param intersectPTs     : (s_ego, s_other)The S value of the intersection NOTE: s is 0 at the start of vec(not whole vec)
        """

        super().__init__(vec, referenceCurv=referenceCurv, v_des_func=v_des_func)
        dt=DT
        v_lim = VLIM
        if(otherwhole is None):
            otherwhole = other_vec
            actInd[1] = 0
        if(whole is None):
            whole = vec.reshape(2,-1).T
            actInd[0] = 0
        ## Align the target vector to make it the same length as the vec
        other_vec = otherwhole[actInd[1]:min(len(otherwhole), actInd[1]+self.vec_len)]
        if(len(other_vec)<self.vec_len):
            other_vec = np.concatenate([other_vec, other_vec[-1] + (other_vec[-1]-other_vec[-2])[None,:] * np.arange(1,self.vec_len - len(other_vec)+1)[:,None] ], axis = 0)
        if(otherDist is not None):
            otherDist = otherDist[0:min(len(otherDist), self.vec_len)]
            if(len(otherDist)<self.vec_len):
                otherDist = np.concatenate([otherDist, np.zeros(self.vec_len - len(otherDist))], axis = 0)

        self.targetx = other_vec[:,0]
        self.targety = other_vec[:,1]
        self.targetvx = self._diffdt(self._const(self.targetx))(self.vec)
        self.targetvy = self._diffdt(self._const(self.targety))(self.vec)
        
        self._dx = self._avrun(self._add(self._const(self.targetx), self._neg(self._x)))
        self._dy = self._avrun(self._add(self._const(self.targety), self._neg(self._y)))
        self._dist = self._distance(self._dx,self._dy)

        self._other_dist = intersectPTs[1] - otherDist 
        self._inter_dist = self._add(self._add(self._neg(self._s),self._const(intersectPTs[0])),self._neg(self._const(self._other_dist)))
        # self._future_inter_dist = self._min(self._add(self._avrun(self._inter_dist),self._scale(1,self._diffdt(self._inter_dist))), self._avrun(self._inter_dist)) # it is just add the current feature wi


        self._dist_safty = self._e(self._dist,-0.08)
        self.features["L1_safe"] = self._L1(self._dist_safty,self._const(0))
        self.features["Linf_safe"] = self._Linf(self._dist_safty,self._const(0))
        # self.features["L2_safe"] = self._L2(self._safty,self._const(0))
        # self.features["L2_futureCol"] = self._L2(self._e(self._neg(self._bounded_future_t())),self._const(0))
        # self.features["L1_inverse_distance"] = self._L1(self._inverse_distance(),self._const(0))
        self._future_distance_safety = self._e(self._future_distance(),p=-0.04)
        self.features["L1_future_distance"] = self._L1(self._future_distance_safety,self._const(0))
        self.features["Linf_future_distance"] = self._Linf(self._future_distance_safety,self._const(0))
        # self.features["L1_inverse_future_distance"] = self._L1(self._normalize(self._const(1),self._limit(self._add(self._future_distance(),self._const(-2)),lwlim=0)),self._const(0))
        # self.features["Linf_inverse_future_distance"] = self._Linf(self._e(self._neg(self._future_distance())),self._const(0))
        self._inter_dist_safty = self._e(self._abs(self._inter_dist),-0.2)
        self.features["L1_inter_dist"] = self._L1(self._inter_dist_safty,self._const(0))
        self.features["Linf_inter_dist"] = self._Linf(self._inter_dist_safty,self._const(0))


        self._future_inter_dist_safty = self._e(self._abs(self._future_inter_dist()),-0.8)
        self.features["L1_future_inter_dist"] = self._L1(self._future_inter_dist_safty,self._const(0))
        self.features["Linf_future_inter_dist"] = self._Linf(self._future_inter_dist_safty,self._const(0))


    def _bounded_future_t(self,bound = [1e-9,1]):
        def retfunc(vec):
            dx = self._dx(vec)
            dy = self._dy(vec)
            dvx = self._add(self._const(self.targetvx), self._neg(self._vx))(vec)
            dvy = self._add(self._const(self.targetvy), self._neg(self._vy))(vec)
            proj = dx * (- dvx) + dy * (- dvy)
            denumer = dvx * dvx + dvy * dvy + 1e-9
            # print("denumerator",denumer)
            future_t = proj/denumer
            return future_t.clip(bound[0],bound[1])
        return retfunc
        # return np.clip(dist/(proj+1e-5),-10,10)  # all the same direction projection will be positive, so the maxlimit is 0

    def _future_distance(self):
        future_t = self._bounded_future_t([1e-9,1])
        return self._distance(
            self._add(self._add(self._avrun(self._x),self._product(future_t,self._vx)),
                        self._neg(self._add(self._avrun(self._const(self.targetx)), self._product(future_t,self._const((self.targetvx)))))),
            self._add(self._add(self._avrun(self._y),self._product(future_t,self._vy)),
                        self._neg(self._add(self._avrun(self._const(self.targety)), self._product(future_t,self._const((self.targetvy)))))))


    def _bounded_future_inter_t(self,bound = [1e-9,1]):
        def retfunc(vec):
            proj = self._avrun(self._inter_dist)(vec)
            denumer = -self._diffdt(self._inter_dist)(vec)
            # print("denumerator",denumer)
            future_t = proj/(denumer + 1e-9)
            return future_t.clip(bound[0],bound[1])
        return retfunc

    def _future_inter_dist(self):
        future_t = self._bounded_future_inter_t([1e-9,1])
        return self._add(self._avrun(self._inter_dist), self._product(future_t,self._diffdt(self._inter_dist)))
        
    def _inverse_distance(self):
        def ret_func(vec):
            threshold  = 15
            distance = self._dist(vec)
            # print(type(distance))
            # print(distance)
            factors = np.ones_like(distance) 
            factors[distance>threshold] *=2
            distance = distance * factors # write in this way because Array box doesnot support assignment
            return 1/distance
        return ret_func
            
INTERSECTLIM = 1 # the distance to judge the path to have intersection
class DoubleCarSRWrapper(DoubleCarSRFeatures):
    """
     This class is a simple to encapsulate the calculation of actInd, and intersectInd for the doubleCarfeatures.
        So for better performance, don't initiate the DoubleCarWrapper over and over again, call the `update` function to save calculation
    """
    def __init__(self, vec, other_vec, whole, otherwhole, actInd, **kwargs):
        actInd = (int(actInd[0]),int(actInd[1]))
        DistanceMat = np.array([[distance(i,j) for j in otherwhole] for i in whole])

        # print("Mindistance:", np.min(DistanceMat))
        # print("len(whole),len(otherwhole)",len(whole),len(otherwhole))

        insectPT = np.nonzero(DistanceMat<INTERSECTLIM) # The index of the intersection point
        try:
            insectPT = (insectPT[0][0],insectPT[1][0])
        except IndexError: # the case where is no pair of points fall in the threshold, find the least distance point instead
            insectPT = np.unravel_index(np.argmin(DistanceMat, axis=None), DistanceMat.shape)
        
        egopath = whole[actInd[0]:insectPT[0]]
        otherpath = otherwhole[actInd[1]:insectPT[1]]
        # path a None in is just to call the lambda function
        egoPTs = np.sum((self._distance(self._diff(lambda x: egopath[:,0]),self._diff(lambda y: egopath[:,1])))(None) )
        otherPTs = np.sum(self._distance(self._diff(lambda x: otherpath[:,0]),self._diff(lambda y: otherpath[:,1]))(None))
        otherDistance = self._cumsum(self._distance(self._diff(lambda x: otherwhole[actInd[1]:,0]),self._diff(lambda y: otherwhole[actInd[1]:,1])))(None)
        self.other_vec = other_vec
        self.whole = whole
        self.otherwhole = otherwhole
        self.actInd = actInd
        self.insectPT = insectPT
        self.egoPTs = egoPTs 
        self.otherPTs = otherPTs 
        self.otherDistance = otherDistance
        super().__init__(vec, other_vec, whole=whole, otherwhole=otherwhole, actInd=actInd, 
                        intersectInd=insectPT, intersectPTs=(self.egoPTs, self.otherPTs), otherDist=otherDistance, **kwargs)

    def update(self,vec):
        super().__init__(vec, self.other_vec, whole=self.whole, otherwhole=self.otherwhole, actInd=self.actInd, 
                        intersectInd=self.insectPT, intersectPTs=(self.egoPTs, self.otherPTs), otherDist=self.otherDistance)


class featureFuncWrapper(Features):
    """
        the class that can define some fancy calculations of features
    """
    def __init__(self, vec, Func, referenceCurv=None, v_des_func=None, **kwargs):
        super().__init__(vec,  referenceCurv=referenceCurv, v_des_func=v_des_func, **kwargs)
        self.Func = Func
        self.features["Func_j_lon"] = lambda x: Func("j_lon")(self._jlon(x), self._const(0)(x))
        self.features["Func_v_des"] = lambda x: Func("v_des")(self._v(x),  self._const(self.v_lim)(x))
        self.features["Func_a_lon"] = lambda x: Func("a_lon")(self._alon(x), self._const(0)(x))
        self.features["Func_a_lat"] = lambda x: Func("a_lat")(self._alat(x), self._const(0)(x))
        self.features["Func_ref_d"]   = lambda x: Func("ref_d")(self._ref_d(x), self._const(0)(x))
        self.features["Func_ref_a_d"] = lambda x: Func("ref_a_d")(self._diffdt(self._ref_d)(x), self._const(0)(x))
        self.features["Func_ref_a_s"] = lambda x: Func("ref_a_s")(self._diff(self._ref_ds)(x), self._const(0)(x))
        self.features["Func_ref_sinphi"] =  lambda x: Func("ref_sinphi")(self._ref_sinphi(x), self._const(0)(x))

class MetaFunc():
    """
        the class that can be called like a `Func` in the `featureFuncWrapper`
            given function f, coefDict M
            call as ftr:str -> \lambda x,y,args. f(x,y,M[ftr])
    """
    def __init__(self, func, coefDict):
        self.func = func
        self.coefDict  = coefDict
    def __call__(self,ftr):
        return lambda x,y : self.func(x,y,*self.coefDict.get(ftr,()))