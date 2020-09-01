"""
use the tensorflow to optimize the factors of the cost function
takes in the set of features and its values
    build the graph according to the number of features and then train it.
"""
import tensorflow as tf
import autograd.numpy as np
import pickle as pkl

from inspect import currentframe, getframeinfo # use this to  output the debug messages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

valuePrecision = tf.float64
# SURROGATEOBJ = True # whether or not to use g^T·g to replace g^T·H^{-1}·g
SURROGATEOBJ = False # whether or not to use g^T·g to replace g^T·H^{-1}·g
NEGFACTORLOSS = False # whether use loss penalize negative factors
NEGFACTORRELAX = 0 # the relax factor in the loss of penalize negative factors
SUM1 = True # whether or not all the factors are sumed to 1
INITLR = 0.01
LRDECAY = 0.999
# LRBETA1=0.5
# LRBETA2=0.9
TOTALEPOCHS = 5000
LOGEPOCH = 100
# SCALECONST = 0
DROPFAILEDVALUES = False # Whether all not to drop the failed values
INIT_WEIGHT = None

def getSetting():
    """
        return a string about all the flags in this file
    """
    return "### LearnFactorSetting:\n\n" + "\n".join(["%s : %s"%(k,str(v)) for k,v in 
       [("SURROGATEOBJ",SURROGATEOBJ),
        ("NEGFACTORLOSS",NEGFACTORLOSS),
        ("**SUM1**",SUM1),
        ("**INITLR**",INITLR),
        ("LRDECAY",LRDECAY),
        ("TOTALEPOCHS",TOTALEPOCHS),
        ("INIT_WEIGHT",INIT_WEIGHT)]])


def LR_controller(epoch):
    # return INITLR / int(np.log(epoch+1)+1)
    # return INITLR * 0.997 ** epoch -> doubleCar1-27.md
    # return INITLR * 0.995 ** epoch # -> doubleCar1-27-2.md
    return INITLR * LRDECAY ** epoch

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

# logger = Logger("./logs")

def buildGraph(feed_symbols):
    """
    symbols is a list of str, each representing a feature
    features is list, each item is an tuple, with items arranged in the same order of symbols
    """
    with tf.variable_scope("main_graph",reuse = tf.AUTO_REUSE):
        num_feature = len(feed_symbols)
        initValue =  INIT_WEIGHT if INIT_WEIGHT is not None else [(1/num_feature if(SUM1) else 1)]*(num_feature)
        weight_vector = tf.get_variable("weight_vector", dtype=valuePrecision, # 21650
                                    initializer=tf.constant(initValue[:num_feature-1], dtype = valuePrecision))  # one less than dim_weights
        weight_vector_PH = tf.placeholder(shape = (num_feature-1,),name='weight',dtype=valuePrecision,)
        weight_vector_assign = tf.assign(weight_vector,weight_vector_PH)

        weight_vector =  tf.debugging.assert_all_finite(weight_vector,"weight_vector is NOT FINITE at weight_vector")
        # weight_vector =  tf.Print(weight_vector,[weight_vector],message = "\nweight_vector")

        dim_weights = weight_vector.shape[0]
        tf_learning_rate = tf.placeholder(name="lr", dtype=valuePrecision)
        # arrange the placeholders in a dict, each value of the dict is a tuple: (g place holder, h place holder)
        PHs = {k:(tf.placeholder(shape = (None),name = "g_"+k, dtype = valuePrecision),
                #   tf.placeholder(shape = (None,None),name = "h_"+k, dtype = valuePrecision)  ) for k in feed_symbols}
                tf.placeholder(shape = (None,None),name = "H_"+k, dtype = valuePrecision)  ) for k in feed_symbols}

        feature_dim = [] # The meaning of each dimension of the weight_vector
        dim_traj = tf.shape(list(PHs.values())[0][0])[0] # the first feature's placeholder , the first item(g), the first dim
        tf_g = tf.zeros(dim_traj,dtype = valuePrecision)  # Gradient
        tf_H = tf.zeros((dim_traj, dim_traj),dtype = valuePrecision)  # Hessian

        for k,(gPH,HPH) in PHs.items():
            # print(dim_weights)
            if(len(feature_dim)==dim_weights): # the last feature, the dim_weights has already been used
                if(SUM1):
                    tfsum = tf.reduce_sum(weight_vector)
                    tf_g += PHs[k][0] * (1. - tfsum)
                    tf_H += PHs[k][1] * (1. - tfsum)
                else:
                    tf_g += PHs[k][0] * 1.
                    tf_H += PHs[k][1] * 1.
                tf_H = tf.debugging.assert_all_finite(tf_H,"tf_H is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))
            else:
                # print(weight_vector.shape,len(feature_dim))
                tf_g += PHs[k][0] * weight_vector[len(feature_dim)] 
                tf_H += PHs[k][1] * weight_vector[len(feature_dim)]
                tf_H = tf.debugging.assert_all_finite(tf_H,"tf_H is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))
            feature_dim.append(k)

        regularizer = .1
        # regularizer = .001
        tf_g = tf.expand_dims(tf_g,1)
        tf_H += regularizer * tf.eye(dim_traj, dtype = valuePrecision)

        # check the validitiy of tf_H
        tf_H = tf.debugging.assert_all_finite(tf_H,"tf_H is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))

        tf_H_inv_g = tf.matrix_solve_ls(tf_H, tf_g, l2_regularizer=regularizer)
        tf_H_inv_g = tf.debugging.assert_all_finite(tf_H_inv_g,"tf_H_inv_g is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))


        # minimize g^T H^-1 g - log |H| TODO: Levine eq. (2) says log |-H| and additionally -d_u log(2Pi)
        # equivalent to maximize -g^T H^-1 g + log |H|

        ####### SCALE DOWN tf_H
        # the idea of solving the numerical problem is to scale the matrix first and then scale it back
        # Note that after scale the det may be 0, so add a small constant on it.
        # scale = tf.cast(tf.reduce_mean(abs(tf_H))+SCALECONST,tf.float64) 
        # scale = tf.svd(tf_H, compute_uv=False)/10
        # scale = tf.Print(scale,[scale],message = "\n###SCALE")
        # tf_H = tf_H / tf.cast(scale,valuePrecision)
        # ##### calculate the average log tf_H and the average tf_H 应该 average log tf_H 控制到0 
        # dbg_tf_H = tf.reduce_mean(tf_H)
        # dbg_log_tf_H = tf.reduce_max(tf.log(abs(tf_H)))

        # tf_det = tf.matrix_determinant(tf_H)+ tf.cast(tf.constant([1e-6]), tf.float64)
        # tf_det = tf.Print(tf_det,[tf_det],  message = "\nSCALED det")
        # ###### SCALE UP tf_H and tf_det
        # tf_H = tf_H *tf.cast(scale,valuePrecision)
        # tf_det = tf_det * tf.cast(scale**tf.cast(dim_traj,tf.float64),valuePrecision)
        # tf_det = tf.Print(tf_det,[tf_det],message = "\nSCALED det")
        # tf_log_det = tf.log(tf_det)

        ####
        # USE SVD to compute the det
        ####
        tf_log_det = tf.math.reduce_sum(tf.log(tf.svd(tf_H, compute_uv=False)))

        # tf_log_det = tf.Print(tf_log_det,[tf_log_det,tf_det, dbg_tf_H, dbg_log_tf_H],message = "log_det,det, meantfH, mean log tfH")
        # tf_log_det = tf.Print(tf_log_det,[tf_log_det],message = "log_det,det, meantfH, mean log tfH")
        tf_log_det = tf.debugging.assert_all_finite(tf_log_det,"tf_log_det is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))
        if(not SURROGATEOBJ):
            irl_loss = tf.matmul(tf.transpose(tf_g), tf_H_inv_g) -  tf_log_det
        else:
            irl_loss = tf.matmul(tf.transpose(tf_g),tf_g)
         
        if(NEGFACTORLOSS):
            irl_loss += tf.reduce_sum(- 0.02* tf.log(NEGFACTORRELAX + weight_vector)) 
            if(SUM1):
                irl_loss += - 0.02* tf.log(1 + NEGFACTORRELAX - tf.reduce_sum(weight_vector))

        irl_loss = tf.debugging.assert_all_finite(irl_loss,"irl_loss is NOT FINITE at line %d"%(getframeinfo(currentframe()).lineno))
        # minimization_problem = tf.train.AdamOptimizer(tf_learning_rate).minimize(irl_loss)
        # tf_opt = tf.train.AdamOptimizer(tf_learning_rate,beta1 = LRBETA1, beta2 = LRBETA2)
        tf_opt = tf.train.GradientDescentOptimizer(tf_learning_rate)
        grads_and_vars = tf_opt.compute_gradients(irl_loss)
        minimization_problem = tf_opt.apply_gradients(grads_and_vars)
        # var_grad = tf.gradients(irl_loss, [weight_vector])[0]
    return feature_dim, (minimization_problem, irl_loss, grads_and_vars,tf_learning_rate,PHs), (weight_vector, weight_vector_assign, weight_vector_PH)
    
def main(symbols, values, feed_symbols):
    feature_dim, buildgraph_problem, buildgraph_weight = buildGraph(feed_symbols)
    minimization_problem, irl_loss, grads_and_vars, tf_learning_rate,PHs = buildgraph_problem
    weight_vector, weight_vector_assign, weight_vector_PH = buildgraph_weight

    np.random.seed(0)
    tf.set_random_seed(0)
    sess = tf.Session()
    sess.__enter__()
    tf.global_variables_initializer().run()
    
    loss_history = []
    variable_history = {k:[] for k in feature_dim}
    grad_history = {k:[] for k in feature_dim} # the history of the gradients of each dimension of theta
    old_weight = sess.run([weight_vector])
    old_weight1 = sess.run([weight_vector])

    for epoch in range(1,TOTALEPOCHS+1):
        losss = []
        batchGrads = []
        DropedList = []
        print("# trajs:" ,len(values))
        for trajind,ftr in enumerate(values): # go through every features, One at a time
            # print(trajind)
            load_dict = {k: v for k,v in zip(symbols,ftr)}

            ftrDict = {}
            for k in feed_symbols:
                ftrDict[PHs[k][0]] = load_dict["g_"+k]
                ftrDict[PHs[k][1]] = load_dict["H_"+k]

            ftrDict[tf_learning_rate] = LR_controller(epoch)
            weights_ = sess.run([weight_vector], feed_dict = ftrDict)

            old_weight1 = old_weight # only go back one step does not work, don't know why
            old_weight = weights_
            # print("old_wight,old_wight1:",old_weight,old_weight1)
            try:
                _, irl_loss_,weights_,grads_vars  = sess.run([minimization_problem, irl_loss, weight_vector, grads_and_vars], 
                                # feed_dict={tf_learning_rate: 0.1}.update(ftrDict))
                                feed_dict = ftrDict)
                # weights_ = sess.run([weight_vector], feed_dict = ftrDict)[0]
            # except:
            #     try:
            #         print("Find safe region I")
            #         sess.run(weight_vector_assign,feed_dict = {weight_vector_PH:old_weight1[0]}) # roll back the weight vector
            #         ftrDict[tf_learning_rate] = ftrDict[tf_learning_rate]/5
            #         _, irl_loss_ = sess.run([minimization_problem, irl_loss, weight_vector], 
            #                         # feed_dict={tf_learning_rate: 0.1}.update(ftrDict))
            #                         feed_dict = ftrDict)
            #         weights_ = sess.run([weight_vector], feed_dict = ftrDict)[0]
            #     except:
            #         try:
            #             print("Find safe region II")
            #             sess.run(weight_vector_assign,feed_dict = {weight_vector_PH:old_weight1[0]})
            #             ftrDict[tf_learning_rate] = ftrDict[tf_learning_rate]/25
            #             _, irl_loss_ = sess.run([minimization_problem, irl_loss, weight_vector], 
            #                             # feed_dict={tf_learning_rate: 0.1}.update(ftrDict))
            #                             feed_dict = ftrDict)
            #             weights_ = sess.run([weight_vector], feed_dict = ftrDict)[0]

                if(not np.isnan(irl_loss_) and not np.isinf(irl_loss_)):
                    losss.append(irl_loss_)
                    # print(grads_vars)
                    grads_ = grads_vars[0][0]
                    batchGrads.append(grads_)

            except KeyboardInterrupt as ex:
                return loss_history, variable_history, grad_history
            except Exception as ex:
                print("####epoch:",epoch,"traj:",trajind,"FAILED, skipping")
                print(ex)
                sess.run(weight_vector_assign,feed_dict = {weight_vector_PH:old_weight1[0]})
                weights_ = old_weight1[0]
                # print("old_wight,old_wight1,current_weights:",old_weight,old_weight1,weights_)
                if(DROPFAILEDVALUES):   
                    DropedList.append(trajind)            
                continue
            if(DROPFAILEDVALUES):
                for ind in DropedList[::-1]:
                    del(values[ind])


            # weights_[0][0] *= cost_a_lat_normalizer / cost_v_normalizer
            # weights_[0][1] *= cost_a_lon_normalizer / cost_v_normalizer

            # for i in range(len(weights_)):
            #     weights_[i] *= load_dict[feature_dim[i]+"_normalizer"] / load_dict[feature_dim[-1]+"_normalizer"]
            # print("loss: " + str(irl_loss_) + ", weights: " + str(weights_))
        
        if(epoch % LOGEPOCH==0):
            print("Epoch:",epoch)
        try:
            loss = np.array(losss).mean()
            btgrad = np.mean(batchGrads,axis = 0)
            loss_history.append(loss)
            # logger.scalar_summary("loss", loss, epoch)
            for i,fstr in enumerate(feature_dim):
                if(i<len(feature_dim)-1):
                    # logger.scalar_summary(fstr,weights_[i],epoch)
                    variable_history[fstr].append(weights_[i])
                    grad_history[fstr].append(btgrad[i])
                else:
                    if(SUM1):
                        # logger.scalar_summary(fstr,1-np.sum(weights_),epoch)
                        variable_history[fstr].append(1-np.sum(weights_))
                    else:
                        # logger.scalar_summary(fstr,1.,epoch)
                        variable_history[fstr].append(1)
            print("loss",loss)
            print("variable:",[variable_history[fstr][-1] for fstr in feature_dim])
            print("grads:",[grad_history[fstr][-1] for fstr in feature_dim[:-1]])
        except Exception as ex:
            print(ex)
            return loss_history, variable_history, grad_history
    return loss_history, variable_history, grad_history
    

def testWeights(symbols, values, feed_symbols,weightList):
    """
        calculate the loss of each weight of the weightList
        return losses
    """
    feature_dim, buildgraph_problem, buildgraph_weight = buildGraph(feed_symbols)
    minimization_problem, irl_loss, grads_and_vars, tf_learning_rate,PHs = buildgraph_problem
    weight_vector, weight_vector_assign, weight_vector_PH = buildgraph_weight
    losses = []
    weights = []
    sess = tf.Session()
    sess.__enter__()
    tf.global_variables_initializer().run()

    num_feature = len(feed_symbols)
    for w in weightList:
        try:
            sess.run(weight_vector_assign,feed_dict = {weight_vector_PH:w[:num_feature-1]})

            epochloss = []
            for trajind,ftr in enumerate(values): # go through every features, One at a time
                # print(trajind)
                load_dict = {k: v for k,v in zip(symbols,ftr)}

                ftrDict = {}
                for k in feed_symbols:
                    ftrDict[PHs[k][0]] = load_dict["g_"+k]
                    ftrDict[PHs[k][1]] = load_dict["H_"+k]

                loss, weight = sess.run([irl_loss, weight_vector],feed_dict = ftrDict)
                epochloss.append(loss[0][0])
            losses.append(np.array(epochloss).mean(axis = 0))
            weight = list(weight)+[(1 - np.sum(weight) if(SUM1) else 1)]
            weights.append(weight)
        except KeyboardInterrupt as ex:
            break
        except Exception as ex:
            print(w,ex)

    return losses,weights

## UTIL Functions

def plotres(loss_history, variable_history, grad_history, doc):
    # Util function to generate the experiment results
    doc.addparagraph("weights:")
    sm =0
    finalVariable = {}
    for k,v in variable_history.items():
        vm = np.array(v[-10:]).mean()
        sm += vm
        doc.addparagraph("{}:{}".format(k,vm))
        finalVariable[k] = vm

    doc.addparagraph("the weights adds to:{}".format(sm))
    doc.addparagraph("loss history")

    x = np.arange(len(loss_history))
    plt.plot(x,loss_history)
    plt.title("loss_history")
    doc.addplt()
    plt.clf()

    half = int(len(x)/2)
    plt.plot(x[half:],loss_history[half:])
    plt.title("loss_history_last_half")
    doc.addplt()
    plt.clf()
    
    doc.addparagraph("variables history")

    for k,v in variable_history.items():
        plt.plot(x,v)
        plt.title(k)
        doc.addplt()
        plt.clf()

    doc.addparagraph("grad Histroy")

    for k,v in grad_history.items():
        plt.plot(v)
        plt.title(k)
        doc.addplt()
        plt.clf()
    return finalVariable

if __name__ == "__main__":
    feed_symbols = [ "L2_a_lat",
                        "L2_v_des",
                        "L2_a_lon"]
    # feed_symbols = [ "cost_a_lat",
    #                     "cost_v",
    #                     "cost_a_lon"]
    dataFile = "TRAJ_US_RD_SR_indep_8-22_Procs_831.pkl"
    # dataFile = "TRAJ_US_RD_SR_indep_8-22_Processed.pkl"
    with open(dataFile,"rb") as f:
        dump_symbols,features = pkl.load(f)
    main(dump_symbols,features,feed_symbols)
    # print(locals()["g_cost_a_lat"])
    # print(g_cost_a_lat)
    # ftrDict[g_cost_a_lat] = None
    # ftrDict[locals()["g_cost_a_lat"]] = None

    # ftrDict = {locals()[k]:None for k in feed_symbols}

