#!/usr/bin/env python3
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import argparse
import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
#from keras.datasets import cifar10
import pickle
from scipy.optimize import dual_annealing # for local search 
# Helper functions
from differential_evolution import differential_evolution
#from scipy.optimize import differential_evolution
import helper


class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(224, 224)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions
        # The below line is revised to keep the probabilities
        network_stats, correct_imgs, self.original_probability = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False, epsilon=0.5, no_stop=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if (confidence[target_class] <= epsilon and not no_stop and
                ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class))):
            return True

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (-1, 1), (-1, 1),(-1, 1)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose, epsilon=epsilon, no_stop=no_stop)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False, disp=True, DE=DE, LS=LS)
        
        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_probs[actual_class] < epsilon and (predicted_class != actual_class)
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x]

    def attack_all_least(self, models, samples=500, pixels=(64,), targeted=False,
                   maxiter=75, popsize=400, verbose=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        results = []
        target = None
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            #img_samples = np.random.choice(valid_imgs, samples)
            img_samples = valid_imgs#[0:samples] ## revised by Yisong 2020.09.06, choose the first samples of images to attack
            #img_samples = valid_imgs * samples ## revised by Yisong 2020.08.29

            for i, img in enumerate(img_samples):
                print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                print('The original confidence:  %f'%self.original_probability[i][self.y_test[i]])
                old_result = [] 
                pixel_count = pixels[0] # the up_pixel_count
                low_pixel_count = 0
                up_pixel_count = pixel_count
                fail_pixel_count = 0
                while(up_pixel_count - low_pixel_count > 1):
                   print('The number of pixels is:', pixel_count)
                   result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose, DE=DE, epsilon=epsilon, LS=LS, no_stop=no_stop)
                   if len(old_result)==0: 
                       old_result = copy.copy(result)
                       if not result[5]: break # fail with the most pixels

                   if result[5]: # success attack
                       old_result = copy.copy(result)
                       up_pixel_count = pixel_count
                   else:
                       low_pixel_count = pixel_count

                   pixel_count = int(low_pixel_count + (up_pixel_count - low_pixel_count)/2)

                model_results.append(old_result)
            results += model_results
            helper.checkpoint(results, targeted)
        return results

    def attack_all(self, models, samples=500, pixels=(1, 3, 5), targeted=False,
                   maxiter=75, popsize=400, verbose=False, DE='DE', epsilon=0.5, LS=0, no_stop=False):
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            #img_samples = np.random.choice(valid_imgs, samples)
            img_samples = valid_imgs#[0:samples] ## revised by Yisong 2020.09.06, choose the first samples of images to attack
            #img_samples = valid_imgs * samples ## revised by Yisong 2020.08.29

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)
                    print('The original confidence:  %f'%self.original_probability[i][self.y_test[i]])
                    for target in targets:
                        if targeted:
                            print('Attacking with target', self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose, DE=DE, epsilon=epsilon, LS=LS, no_stop=no_stop)
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results

def ultra_attack(model, samples, target):
    ''' one-pixel attacking each sample in samples by reversing the point
    '''
    bounds = [(-1, 1), (-1, 1),(-1, 1)]
    limits = np.array(bounds, dtype='float').T    
    __scale_arg1 = 0.5 * (limits[0] + limits[1])
    __scale_arg2 = np.fabs(limits[0] - limits[1])

    def _scale_parameters(trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return __scale_arg1 + (trial - 0.5) * __scale_arg2

    def _unscale_parameters(parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - __scale_arg1) / __scale_arg2 + 0.5

    def _farmost(parameters):
        t = np.zeros((3))
        for i in range(3):
            if parameters[i] < 0: 
                t[i] = 1.
            else:
                t[i] = -1.
        return t

    for i in range(len(samples)):
        print('starting attack the %d sample'%i)
        best_prob = 1.
        type = target[i]
        
        for x in range(224):
            for y in range(224):
                rgb = samples[i][x,y]
                samples[i][x,y] = _farmost(rgb) #_scale_parameters( 1.0 - _unscale_parameters(samples[i][x,y]) )
                prob =  model.predict(samples[i].reshape(1,224,224,3))[0][type] 
                samples[i][x,y] = rgb
                if best_prob > prob: 
                    best_prob, best_x, best_y, = prob, x, y 
                if best_prob < 0.5: break
            if best_prob < 0.5: break   
        print('best solution x=%d, y=%d with probability=%f'%(best_x,best_y, best_prob))
        print(samples[i][best_x,best_y])

def get_IDs_RA(up, low, pdata,type=1, top_k=10):
#   get the ids to do repeat attack
    _f = open(pdata,'rb')
    _data = pickle.load(_f)
    _f.close()
    IDs = [] 
    ## choise the top k cdiff 
    dlen = len(_data)
    t_data = np.zeros((dlen,4)) # to keep ID prior-conf attacked-conf cdiff
    for i in range(dlen):
        t_data[i][0], t_data[i][1], t_data[i][2], t_data[i][3] = _data[i][2], _data[i][7][type], _data[i][8][type], _data[i][6] 
    T_data = t_data[np.argsort(t_data[:,3])][ dlen - top_k:] 
    return T_data

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Attack models on DR, CXR, DERM') 
    parser.add_argument('--source', default='dr', help='The source of being attacked[cxr/dr/derm].')
    parser.add_argument('--model', default='./model/dr/wb_model.h5', help='The trained model to be attacked.')
    parser.add_argument('--pixels', nargs='+', default=(1,), type=int,
                        help='The number of pixels that can be perturbed.')
    parser.add_argument('--maxiter', default=35, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=8, type=int, 
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--data', default='dr_pos_data_10.npy', help='The data file.')
    parser.add_argument('--type', default=1, type=int, help='The type of attacked samples, 0: negative, 1: positive.')
    parser.add_argument('--LP', action='store_true',  help='Compute the least pixels to sucessfully attack.')
    parser.add_argument('--save', default='results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--DE', default='DE', help='The differential evolution algorithm: DE, SHADE, EBLSHADE.')
    parser.add_argument('--epsilon', default=0.5, type=float, help='The confidence threshold.')
    parser.add_argument('--no_stop', action='store_true', help='Do not stop when find a solution.')
    parser.add_argument('--LS', default=0, type=int, help='Do local searching: (0: no local search; 1: local search at the end; 2: local search at each better solution; 4: local search at each better solution and replace the best with the the new one if it is better than the last best one.).')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    # The following paremeters are for repeating attack   
    parser.add_argument('--RA', default=1, type=int, help='The repeating times of repeating attacks')
    parser.add_argument('--RAn', default=10, type=int, help='The number samples of repeating attack')
    parser.add_argument('--low', default=0.1, type=float, help='The confidence after first attack')
    parser.add_argument('--up', default=0.9, type=float, help='The confidence before first attack')
    parser.add_argument('--pickle', type=str, help='The results pickle file')

    args = parser.parse_args() 

    class_names = ['positive','negative']
    model = load_model('./model/'+args.source + '/wb_model.h5')
    true_label = np.load('./data/'+args.source + '/val_test_y.npy')
    predictions = np.load('./data/'+args.source +'/winning_model_preds.npy')
    data = np.load('./data/'+args.source + '/val_test_x_preprocess.npy')
    both_true = (true_label[:,args.type] >= 0.5) & (predictions[:,args.type] >= 0.5)
    if args.samples == -1 or args.samples > sum(both_true):
       args.samples = sum(both_true)
    test = data[both_true][0:args.samples], np.array([[args.type]]*args.samples)

    if args.RA > 1:
        _ids = get_IDs_RA(args.up, args.low, 'results/' + args.source + '/' + args.pickle, args.type, top_k=args.RAn)
        _test = np.zeros((len(_ids),224,224,3))    
        for i in range(min(len(_ids), args.RAn)): 
           print(_ids[i])
           _test[i] = test[0][ int(_ids[i][0]) ]
        test = _test,  np.array([[args.type]] * len(_ids))
        RA_results = []
        attacker = PixelAttacker([model], test, class_names)
        for i in range(args.RA):
           results = attacker.attack_all([model], samples=len(_ids), pixels=args.pixels, targeted=args.targeted, \
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose, DE=args.DE,  \
                                  epsilon=args.epsilon, LS=args.LS, no_stop=args.no_stop)
           RA_results += results
        columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation']
        results_table = pd.DataFrame(RA_results, columns=columns)
        print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success']])
        print('Saving to', args.save)
        with open(args.save, 'wb') as file:
           pickle.dump(RA_results, file)
        
        ## draw the figures with annotated attack points
        # for each sample 
        for i in range(min(len(_ids), args.RAn)):
           _path = './results/' + args.source + '/'
           _file_name = _path + 'RA-' +str(i) + '-' + str( int(_ids[i][0]) ) + '-' + str(args.type) + '.png'

           # the original image
           fix,ax = plt.subplots(1)
           ax.imshow(_test[i], origin='lower')
           plt.title(' Prior conf=' + format(RA_results[i][7][args.type],'.4f'))
           plt.xlabel('ID='+str(int(_ids[i][0])))
           plt.savefig(_path + 'RA-ori-' + str(i) + '-' + str( int(_ids[i][0]) ) + '-' + str(args.type) + '.png')
           plt.close('all')

           # the attacked image
           fig,ax = plt.subplots(1)
           ax.set_aspect('equal')
           _cdiff = 0.0
           
           # get the attacking pixels 
           ## draw circles of repeated attacks 
           for j in range(args.RA):
              _px, _py = int(RA_results[j * args.RAn + i][9][0]), int(RA_results[j * args.RAn + i][9][1])
              print(j, _px, _py)
              _cdiff += RA_results[j * args.RAn + i][6] 
              plt.scatter(_px, _py, color='', edgecolor='b', marker='o', alpha=.5, s=150)
              _test[i][_py,_px] = RA_results[j * args.RAn + i][9][2], RA_results[j * args.RAn + i][9][3], RA_results[j * args.RAn + i][9][4] 
  
           _title = ' cdiff=' + format(_cdiff/args.RA,'.4f')
           plt.title(_title)
           plt.xlabel('ID='+str(int(_ids[i][0])))
           ax.imshow(_test[i], origin='lower')
           plt.savefig(_file_name)
           plt.close('all')
 
        os._exit(0)          
    

    attacker = PixelAttacker([model], test, class_names)

    print('Starting attack')
   
    if args.LP:
        results = attacker.attack_all_least([model], samples=args.samples, pixels=args.pixels, targeted=args.targeted,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose, DE=args.DE, 
                                  epsilon=args.epsilon, LS=args.LS, no_stop=args.no_stop)
    else:
        results = attacker.attack_all([model], samples=args.samples, pixels=args.pixels, targeted=args.targeted,
                                  maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose, DE=args.DE, 
                                  epsilon=args.epsilon, LS=args.LS, no_stop=args.no_stop)

    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs', 'perturbation']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success']])

    print('Saving to', args.save)
    with open(args.save, 'wb') as file:
        pickle.dump(results, file)
    
