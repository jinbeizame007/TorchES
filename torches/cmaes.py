import torch
import numpy as np
import cma

from collections import OrderedDict


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


class ES:
    def __init__(self, model, es):
        self.model = model
        self.es = es
        self.solutions = None
    
    @property
    def result(self):
        return self.es.result

    # get list of new solutions (models)
    def ask(self):
        models_params = self.es.ask()
        self.solutions = models_params.copy()
        models = self.convert_params_to_models(models_params)
        return models
    
    def tell(self, fitness):
        fitness = np.array(fitness)
        self.es.tell(self.solutions, fitness) # convert minimizer to maximizer.

    # set params to model
    def convert_params_to_models(self, models_params):
        models = []
        for model_params in models_params:
            # prepare state_dict
            model_params = np.split(model_params, self.sections)
            state_dict = OrderedDict()
            for key, params, shape in zip(self.keys, model_params, self.shapes):
                state_dict[key] = torch.FloatTensor(params.reshape(shape))

            # set state_dict to model
            model = type(self.model)() # get a new instance
            model.load_state_dict(state_dict)
            models.append(model)
        return models
    
    # set sizes, num_params_per_group, num_params and keys
    def set_model_info(self, model):
        self.shapes = []
        for params in model.parameters():
            self.shapes.append(list(params.size()))
        num_params_per_group = [np.prod(shape) for shape in self.shapes]
        self.sections = [sum(num_params_per_group[:i]) for i in range(1, len(num_params_per_group))]
        self.num_params = sum(num_params_per_group)
        self.keys = list(model.state_dict().keys())


class CMAES(ES):
    def __init__(self, model, 
                 sigma_init=1.0,
                 popsize=64,
                 weight_decay=0.01):

        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.set_model_info(model)

        es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                        self.sigma_init,
                                        {'popsize': self.popsize,
                                            })
        super(CMAES, self).__init__(model, es)
    
    def tell(self, fitness):
        fitness = np.array(fitness)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            fitness += l2_decay
        self.es.tell(self.solutions, fitness)

