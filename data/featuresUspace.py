"""
Some implementation where the action space is not x,y
"""        

import sys
import os
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath)
from features import *
import autograd.numpy as np
from autograd import grad, jacobian

class UspaceFtr:
    """
        The base class of the Uspace features. 
            This class use a subclass of feature class as its attribute, and works as a wrapper around that class.
            Translates the input vector in action space to xy space, and process the gradients to the gradient in the action space
    """
    def __init__(self,Ftr,vec,shortCut = True,**kwargs):
        self.Ftrtype = Ftr
        self.vec_len = int(vec.shape[0] / 2)
        self.kwargs = kwargs
        self.ftr = Ftr(vec,**kwargs)
        self.hasAttrUpdate = hasattr(self.ftr,"update")
        self.Jspace = None # the jacobian matrix: dxy/dds
        # Don't uncomment this, the cache is judged by attribute error
        ## self.Uvec = None # the vector in the U space, should in theory equal to the result of `spaceTransFromXY`
                         # the cache of the property `self.spacevector`
        
        # the features that are more convienent to be computed in self space
        self.features = {}
        self.shortCut = shortCut

    def spaceTransFromXY(self,vec):
        """
            Input is the xyvec and return the vector in Uspace
        """
        raise NotImplementedError

    def spaceTransToXY(self,vec):
        
        raise NotImplementedError

    def update(self,spacevec):
        vec = self.spaceTransToXY(spacevec)
        self.Uvec = spacevec
        self.Jspace = None
        if(self.hasAttrUpdate):
            self.ftr.update(vec) # if the feature class have update, then call the update to save computation
        else:
            self.ftr = self.Ftrtype(vec,**self.kwargs)


    def spaceJacob(self):
        raise NotImplementedError


    def featureGradJacobNormalizer(self,feature,singulervaluenorm = SINGULARVLAUENORM, computeHessian = True):
        if(self.shortCut and feature in self.features.keys()):
            featurefun = self.features[feature]
            g = jacobian(featurefun)(self.flatToVec(self.spacevec))[:]
            if(computeHessian):
                H = jacobian(grad(featurefun))(self.flatToVec(self.spacevec))[:,:]
                assert( not np.isnan(H).any() )
            else:
                H = 1
            if(computeHessian and singulervaluenorm):
                normalizer = np.linalg.svd(H)[1][0]
            else:
                normalizer = 1
            g,H = g/normalizer, H/normalizer        
            return g,H,normalizer
        else:
            g, H, normalizer = self.ftr.featureGradJacobNormalizer(feature,singulervaluenorm, computeHessian)
            J = self.spaceJacob()
            if(computeHessian):
                return g@J, J.T @ H @ J, normalizer
            else:
                return g@J, None, normalizer

    def arangeToArray(self, vec= None):
        raise NotImplementedError

    def flatToVec(self, vec= None):
        raise NotImplementedError

    def featureValue(self, feature, vec = None):
        if(self.shortCut and feature in self.features.keys()):
            featurefun = self.features[feature]
            forward = featurefun(self.flatToVec(self.spacevec))
            return forward
        else:
            featurefun = self.ftr.features[feature]
            if(vec is None):
                vec = self.ftr.vec
            else:
                vec =self.spaceTransToXY(vec)
            forward = featurefun(vec)
            return forward


    @property
    def spacevec(self):
        """
        The real value is called Uvec
        """
        try:
            return self.Uvec
        except AttributeError:
            self.Uvec = self.spaceTransFromXY()
            return self.Uvec

class SDspaceFtr(UspaceFtr):
    """
        Change the space from xy space to DS space(traveled distance and diviation),  where the reference of the S and D is the traj itself.
    """
    def __init__(self, Ftr, vec, baseVector = None,shortCut = True,**kwargs):
        # the reference vector is at best the whole trajectory of the traj of interaction

        if(baseVector is None):
            baseVector = vec.reshape(2,-1).T
        self.baseVector = baseVector
        super().__init__(Ftr, vec,shortCut = shortCut,**kwargs)

        self.refInfocache = None
        self._s = lambda vec : vec[:self.vec_len]
        self._d = lambda vec : vec[self.vec_len:]
        self._vlon = self.ftr._diffdt(self._s)
        self._alon = self.ftr._diffdt(self._vlon)
        self._jlon = self.ftr._diffdt(self._alon)
        self._base_d = LazyFunc(self._base_d_)
        self.features = {"L2_a_lon":self.ftr._L2(self._alon,self.ftr._const(0)),
                        "L1_a_lon":self.ftr._L1(self._alon,self.ftr._const(0)),
                        "Linf_a_lon":self.ftr._Linf(self._alon,self.ftr._const(0)),

                        "L2_j_lon":self.ftr._L2(self._jlon,self.ftr._const(0)),
                        "L1_j_lon":self.ftr._L1(self._jlon,self.ftr._const(0)),
                        "Linf_j_lon":self.ftr._Linf(self._jlon,self.ftr._const(0))} 
        try:
            self.features["Func_a_lon"] = lambda x: self.ftr.Func("a_lon")(
                self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self._alon)))))))
                # self._alon
                    (x),self.ftr._const(0)(x))
            self.features["Func_j_lon"] = lambda x: self.ftr.Func("j_lon")(
                self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self.ftr._avrun(self._jlon)))))))
                # self._alon
                    (x),self.ftr._const(0)(x))
        except  Exception as ex:
            print("Exception:",str(ex))
            raise ex
        
    def arangeToArray(self, vec= None):
        """
        This is the inter conversion in the U space between vectorized and array
        """
        if(vec is None):
            return self.spacevec
        return vec.reshape(2,-1).T

    def flatToVec(self, vec = None):
        if(vec is None):
            vec = self.spacevec
        return vec.T.reshape(-1)


    def spaceTransFromXY(self,vec=None):
        # WARN: If vec is not the vec of ftr, it should be realigned, but currently not
        # WARN: The Implementation is different from the calculation of in features.py
        #       The ref_s in feature.py is kind of approaximation inorder to have gradient 
        if(vec is None):
            vec = self.ftr.vec
        # init_proj = np.array([self.ftr.refcurv[0],self.ftr.refcurv[self.vec_len]])
        # init_ind = self.ftr.refinds[0]

        basevec = self.baseVector.T.reshape(-1)
        (refx, refy, refvx, refvy, refs, refv,refds) = self.refInfo()
        # refs = np.cumsum(np.concatenate([[0],refds],axis = 0))
        
        
        refcurv = self.basecurv.reshape(2,-1).T
        refinds = self.baseinds

        s = refs[refinds] + np.linalg.norm(self.baseVector[refinds] - refcurv,axis = 1)
        return np.concatenate([s[:,None], self._base_d(vec)[:,None]],axis = 1)

    @property
    def basecurv(self):
        try:
            return self.basecurvValue
        except AttributeError:
            self.basecurvValue, self.baseindsValue = self._basecurv(self.ftr.vec)
            return self.basecurvValue

    @property
    def baseinds(self):
        try:
            return self.baseindsValue
        except AttributeError:
            self.basecurvValue, self.baseindsValue = self._basecurv(self.ftr.vec)
            return self.baseindsValue

    @property 
    def baseInfo_rdxrdy(self):
        try:
            return self.baseInfo_rdxrdyValue
        except AttributeError:
            rx = self.basecurv[:self.vec_len]
            ry = self.basecurv[self.vec_len:]
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
            self.baseInfo_rdxrdyValue  = (rdx,rdy)
            return self.baseInfo_rdxrdy

    def _basecurv(self,vec):
        """
        the input vec is a list contains [x,....x, y,.... y]
        """
        # print("vecshape",vec.shape)
        # point_vec = np.concatenate((vec[:self.vec_len,None],vec[self.vec_len:,None]),axis = 1)
        point_vec = vec.reshape(2,-1).T
        # cv_ind = curveMatch(self.baseerenceCurv, point_vec)
        # print(point_vec.shape)
        basecurv, baseinds = closestPoint(self.baseVector, point_vec)
        # basecurv = np.concatenate((basecurv[:,0],basecurv[:,1]), axis = 0)
        basecurv = basecurv.T.reshape(-1)
        return basecurv, baseinds

    def _base_d_(self):
        """
        return the function to calculate the latter deviation to base line
        """
        rdx,rdy = self.baseInfo_rdxrdy
        # assert(not np.any((rdx==0)*(rdy==0)))
        def ret_func(vec):
            d = vec - self.basecurv
            dx = d[:self.vec_len]
            dy = d[self.vec_len:]
            # return (dx**2 + dy**2+1e-9)**0.5
            return (rdx * dy - rdy * dx)/np.sqrt(rdx*rdx + rdy*rdy)
        return ret_func


    def refInfo(self):
        # Hack the ftr to do the calculation
        if(self.refInfocache is None):
            refvec = self.baseVector.T.reshape(-1)
            tmp = self.ftr.vec_len
            self.ftr.vec_len = len(self.baseVector)
            refx = self.ftr._x(refvec)
            refy = self.ftr._y(refvec)
            refvx = self.ftr._vx(refvec)
            refvy = self.ftr._vy(refvec)
            refs = self.ftr._s(refvec)
            refv = self.ftr._v(refvec)
            refds = self.ftr._ds(refvec)
            self.ftr.vec_len = tmp
            self.refInfocache = (refx, refy, refvx, refvy, refs, refv,refds)
        return self.refInfocache

    def spaceTransToXY(self,DS = None):
        if(DS is None): # use the xy saved in self.ftr
            return self.ftr.vec
        svec = DS[:,0]
        dvec = DS[:,1]  
        (refx, refy, refvx, refvy, refs, refv,refds) = self.refInfo()
        
        inds = [np.max(np.where(refs<=s)) for s in svec] # assume the reference line goes straight after the last point
        
        projs = svec - refs[inds]
        projox = refx[inds] # the x coordinate of the origin of projection
        projoy = refy[inds]
        projvx = refvx[np.clip(inds,None,len(refvx)-1)]
        projvy = refvy[np.clip(inds,None,len(refvy)-1)]
        projv = refv[np.clip(inds,None,len(refv)-1)]
        x = projox + (projs * projvx - dvec * projvy) / projv
        y = projoy + (projs * projvy + dvec * projvx) / projv
        return np.concatenate([x,y],axis = 0)


    def spaceJacob(self):
        def T(DSvec):
            vec = self.arangeToArray(DSvec)
            xyvec = self.spaceTransToXY(vec)
            return xyvec
        if(self.Jspace is None):
            self.Jspace = jacobian(T)(self.flatToVec(self.spacevec)) # this operation is quite expensive, use this as a cache
        return self.Jspace
    

# class VsVdSpaceFtr(UspaceFtr):
#     """
#     The action space is the velocity and the velocity of diviation.
#     """
    



class ACCSPACEDoubleCarSRFeatures(DoubleCarSRFeatures):
    """
    The feature class where the action space is the elongation acceleration of each traj point
        (trajectory points is sampled according to distance).
    """
    def __init__(self, vec, other_vec, dt=DT, v_lim=11.17, referenceCurv=None, v_des_func=None,sbSampleRate = 1):
        """
            The init method initiate the base class and change the function pointers `self._x` and `self._y`
            Naming convension:  tb: time batted: The sample points with even time difference
                                sb: distance batted: The sample points with even distance difference

        """
        super().__init__(vec, other_vec, dt=dt, v_lim=v_lim, referenceCurv=referenceCurv, v_des_func=v_des_func)
        self.sbSampleRate = sbSampleRate
        
        # calculates the states of sb
        sb_t, sb_v = sbStates()
        

        sb_vec = np.concatenate([sb_t[:,None], sb_v[:,None],],axis = 1).reshape(-1)
        # calculates the features of sb

        # self._accprofile = np.concatenate([np.array([0]),self._alon(self.vec)],axis = 0)
        self._accprofile = self._alon(self.vec)
        self._dtvprofile = self._dtvprofile_() # [:,0] are dts and [:,1] are v at each acc point
        self._dtprofile = lambda vec: self._dtvprofile(vec)[:,0]
        self._vprofile = lambda vec: self._dtvprofile(vec)[:,1]
        self._tprofile = self._cumsum(self._dtprofile)
        # self._x,self._y = self.

    def sbTrans(self,t,v,a):
        """
        calculates the linearlized instance of the dynamics given the state t,v of a distance batted point.
            The dynamics is: states: (t,v)
                t1 = t0 + 2*ds/(v0+v1)
                v1 = sqrt(v0^2 + 2*a0*ds)
        returns the instance of dyanmics A and B matrix of that point
        """
        ds = self.sbSampleRate
        sqrt_tmp = np.sqrt(v**2 + 2 * a + ds)
        A = np.array([
            [1, (-2*ds(1 + v/sqrt_tmp))/((v+sqrt_tmp)**2)],
            [0, v/sqrt_tmp]
        ]) # 2 x 2 mat
        B = np.array([
            [(-2*ds( ds/sqrt_tmp))/((v+sqrt_tmp)**2)],
            [ds/sqrt_tmp]
        ]) # 2 x 1 mat
        return A,B

    def sbPoints(self):
        """
        calculate the distance batted (sb) states from the tb points. Using the features calculated in the base class.
        return 
            traj_ind 
            t vector
            v vector of each sbpoints
        """

        # The first sb sample point is get from the second tb sampled point, to calculate the init v0
        S = self._s(self.vec)[1:] # (vec_len-1 x 1) the first element is 0
        S = S - S[1]
        X = self._x(self.vec)[1:] # (vec_len-1 x 1) 
        Y = self._y(self.vec)[1:] # (vec_len-1 x 1) 

        traj_ind = [0]
        traj_s = [0]
        xypoints = [(X[0], Y[0])]
        nextDis = self.sbSampleRate
        i = 0
        while (i<len(S)):
            s = S[i]
            nextDis_ = nextDis # the nextDis used in if condition
            if(nextDis_ <= s):
                traj_ind.append(i-1)
                prop = (nextDis-S[i-1])/(S[i] - S[i-1])
                xypoints.append((prop*X[i] + (1-prop)*X[i-1], prop*Y[i] + (1-prop)*Y[i-1]))
                traj_s.append(nextDis)
                nextDis += self.sbSampleRate
            if(nextDis_ >= s):
                i += 1
        return traj_ind, xypoints

    def sbStates(self):
        """
        return a functions calculates dt,v (profilevec_len x 2) [:,0] are dts and [:,1] are v at each acc point
        """
        # acc_profile -> dtdv
        ds = self.sbSampleRate
        v0 = self._v(self.vec)[0]
        acc = self._alon(self.vec)
        # def ret_func(acc):
        v = np.zeros(len(acc)+1) # note that v are corresponding to each acc profile, so the last v is not considered
        dt = np.zeros(len(acc)+1)
        v[0] = v0
        
        for i,a in enumerate(acc):
            v[i+1] = np.sqrt(v[i]^2 + 2*a*ds)
            dt[i+1] = 2*ds/(v[i]+v[i+1])
        t = np.cumsum(dt)
        return t,v

    

    # def trajSamples(self):
    #     """
    #     calculates the evenly distannce distributed points used as points to hold acclerations
    #         return: traj_ind: the indexes represents which section in the time sampled xy vector the time belongs
    #               : xy_points: each corresponds to a traj_ind
    #     """
    #     S = self._s(self.vec) # (vec_len x 1) the first element is 0
    #     X = self._x(self.vec) # (vec_len x 1) the first element is 0
    #     Y = self._y(self.vec) # (vec_len x 1) the first element is 0
    #     traj_ind = [0]
    #     traj_s = [0]
    #     xypoints = [(self.vec[0], self.vec[self.vec_len])]
    #     nextDis = self.sbSampleRate
    #     i = 0
    #     while (i<len(S)):
    #         s = S[i]
    #         nextDis_ = nextDis # the nextDis used in if condition
    #         if(nextDis_ <= s):
    #             traj_ind.append(i-1)
    #             prop = (nextDis-S[i-1])/(S[i] - S[i-1])
    #             xypoints.append((prop*X[i] + (1-prop)*X[i-1], prop*Y[i] + (1-prop)*Y[i-1]))
    #             traj_s.append(nextDis)
    #             nextDis += self.sbSampleRate
    #         if(nextDis_ >= s):
    #             i += 1
    #     return traj_ind, xypoints,traj_s

    # def trajSample2TimeSample(self):
    #     """
    #        return functions that calculates the xy from the accleration profile is used to replace self._x and self._y
    #     """
    #     traj_ind, traj_xy, traj_s = self.trajSamples(self.sbSampleRate)
    #     S = self._s(self.vec)
    #     v0 = self._v(self.vec)[0]
    #     vs = [v0]   # the v at each acc sample points
    #     dts = []    # The t spend between every two acc sample points

    #     tprofile = self._tprofile
    #     prof_ind = np.clip([np.min(np.where(traj_ind>=i))-1 for i in range(self.vec_len)],0,None) # sample the points with even dt
    #     vx = self._vx(self.vec)
    #     vy = self._vy(self.vec)
    #     xy = np.concatenate([self.vec[:self.vec_len ,None],self.vec[self.vec_len :,None]],axis = 1)

    #     follows_prof_pt = [traj_s[j] >= S[i] for i,j in enumerate(prof_ind) ]
    #     traj_xy0 = np.array([(traj_xy[j] if follows_prof_pt[i] else xy[i-1]) 
    #                         for i,j in enumerate(prof_ind)]) # the xy points at each of the acc_prof, while it is indexed according to time sampled traj
        
    #     xfunc = self._add(self._const(), self._product(vx,)  )
    #     return xfunc, yfunc


