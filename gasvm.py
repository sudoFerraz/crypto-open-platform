from sklearn import SVC
import numpy

class svc_Model(object):

    def __init__(self):
        self.model = self.create_Model()

    def create_Model():
        model = SVC(C=70, kernel='rbf', degree=3, gamma='auto', \
            coef0=0.0, probability=False, shrinking=True, tol=0.001,\
                decision_function_shape = 'ovr', random_state=42)
        return model

    def fit_Model(x_train, y_train):
        self.model.fit(x_train, y_train.values.ravel())

    def get_Accuracy(x_test, y_test):
        acc = self.model.score(x_test, y_test)
        return acc

class genetic_Algorithm(object):
    def __init__():
        self.retain = 0.25,
        self.random_select = 0.1
        self.mutation = 0.1
        self.population = 20
        self.generations = 20
        self.x_test = []
        self.x_train = []
        self.y_test = []
        self.y_train = []
        self.model = []
        self.actual_generation = 0
        self.best_Score = 0
        best_Individual = []

    def data_Setup(self, x_test, x_train, y_test, y_train, model):
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train
        self.model = model

    def create_Individual(self):
        individual = []
        for i in range(0, 61):
            feature = numpy.random.choice(numpy.arange(0,2), p=[0.5, 0.5])
            individual.append(feature)
        return individual

    def create_Population(self):
        population = []
        for i in range(0, self.population):
            population.append(self.create_Individual())
        return population

    def train_and_Score(self, individual):
        n_x_train = self.x_train.copy()
        n_x_test = self.x_test.copy()
        index = 0
        cols = []
        for i in individual:
            if i == 0:
                cols.append(index)
                index = index + 1

        n_x_train = n_x_train.drop(n_x_train.columns[cols], axis=1)
        n_x_test = n_x_test.drop(n_x_test.colums[cols], axis=1)

        self.model.fit(n_x_train, y_train.values.ravel())
        acc = model.score(n_x_test, y_test)
        return acc

    def mutate(self, individual):
        index = 0
        for i in individuo:
            mutate = numpy.random.choice(numpy.arange(0,2), p=[0.95, 0.05])
            if mutate == 1:
                if i == 0:
                    individual[index] = 1
            index = index + 1
        return individual

    def breed(self, father1, father2):
        son = []
        for i in range(0, (math.ceil((len(father1)*0.5)))):
            son.append(father1[i])
        for i in range((math.ceil((len(father1)*0.5))), len(father1)):
            son.append(father2[i])
        son = self.mutate(son)
        return son

    def evolve(self, population):
        score_Individuals = []
        for individual in population:
            score = self.train_and_Score(individual)
            score_Individuals.append(score)
        best_Fit = numpy.argsort(scores_Individuals)
        best_Fit = list(best_Fit)
        new_Generation = []
        for i in range(0, (len(best_fit))):
            if i == 19:
                son = self.breed(population[best_Fit[i]], \
                    population[best_Fit[i-1]])
            else:
                son = son.breed(population[best_Fit[i]], \
                    population[best_fit[i+1]])
            new_Generation.append(filho)
        self.actual_generation = self.actual_generation + 1
        if scores_Individuals[best_Fit[0]] > self.best_Score:
            self.best_Score = score_Individuals[best_Fit[0]]
            self.best_Individual = population[best_Fit[0]]
        while self.actual_generation < 20:
            self.evolve(new_Generation)
        return new_Generation



    





        