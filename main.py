import numpy as np
import pickle
import random
import nltk
from scipy import sparse
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from collections import Counter
import datetime

nltk.download('punkt')
nltk.download('wordnet')

nltk.download('stopwords')

SUBMISSION_HEADER_CSV = "Id,Category"
IS_TEST = False
STOP_WORDS_LIST = stopwords.words('english')


class RandomPrediction:
    def __init__(self, classes):
        self.classes = classes

    def train(self, train_data):
        pass

    def compute_predictions(self, test_data):
        test_predictions = []

        for i in range(len(test_data)):
            test_predictions.append([i, self.classes[random.randint(0, len(self.classes)-1)]])
        generate_submission_csv(test_predictions, "random")


class NaiveBayesSacMots:
    def __init__(self, classes):
        self.classes = classes

    def train(self, train_data):

        self.priorProbabilityEachClass = np.zeros(len(self.classes))
        #On compte le nombre de fois que le subreddit apparait
        #ATTENTION: self.classes contient deja une occurence de chaque subreddit
        for i in range(len(self.classes)):
            self.priorProbabilityEachClass[i] = train_data[1].count(self.classes[i])
        #On obtient la probabilité de chaque subreddit apparaisse dans notre train data set
        #Ie les priors
        self.priorProbabilityEachClass = np.divide(self.priorProbabilityEachClass, len(train_data[1]))

        #On raffine les commentaire pour avoir des mots clés
        self.train_inputs = data_preprocessing(train_data[0])

        #self.distributionArray pour compter le les mots les plus populaire pour chaque classe
        self.distributionArray = np.zeros(len(self.classes), dtype=np.object)
        for i in range(len(self.train_inputs)):
            #Pour sassurer que les indices de self.classes et self.distrubutionArray corresponde
            j = np.where(self.classes == train_data[1][i])[0][0]
            if self.distributionArray[j] == 0:
                self.distributionArray[j] = Counter(self.train_inputs[i])
            else:
                self.distributionArray[j] += Counter(self.train_inputs[i])

        sum_words_class = np.zeros(len(self.classes))
        #On somme le nombre de mot dans chaque subreddit. Par example 'nba', 560 mots
        for i in range(len(self.classes)):
            sum_words_class[i] = sum(self.distributionArray[i].values())
        #On calcule la probabilité de chaque mot dans les subreddit respectif dans distributionArray
        for i in range(len(self.classes)):
            self.distributionArray[i] = {key: value / sum_words_class[i] for key, value in self.distributionArray[i].items()}

        #outil obtenu apres entrainement:
        # 1) self.priorProbabilityEachClass = probabilité de chaque classe. Par exemple, r/nba, 367/3000 sil y a 367 commentaire tirer du subreddit de nba
        #    Cest la meme taille et les meme indice que self.classes
        #
        # 2) self.distributionArray = liste de liste de probabilié de chaque mot (en format liste) dans les subreddit respectif
        #    Par exemple,  r/nba, 'thompson', 37/300, 'basketball', 27/300
        #    Cest la meme taille et les meme indice que self.classes

    def compute_predictions(self, test_data):
        #Encore une fois on raffine les données et on instancie un tableau de prediction
        preprocessed_test_data = data_preprocessing(test_data)
        prediction_array = np.ones((len(preprocessed_test_data), len(self.classes)))

        #Pour chaque element de notre jeux de donnée de test,
        for i in range(len(preprocessed_test_data)):
            #Pour chaque element de nos classes
            for j in range(len(self.classes)):
                #la probabilite qu'un commentaire i fasse partie de la classe j
                prediction_array[i][j] = np.prod(np.vectorize(self.distributionArray[j].get)(preprocessed_test_data[i], 0))
        #nous sort la plus grand la classe a qui le commentaire a la plus grande probabilite dappartenir
        test_probabilities = np.argmax(prediction_array, axis=1) #TODO verifier si pour valeur de argmax identique,
        # prendre l'index random de donne pas de meilleurs resultats que toujours prendre le premier?

        test_predictions = []

        test_label = []

        for i in range(len(preprocessed_test_data)):
            if test_probabilities[i] == 0:
                test_predictions.append([i, self.classes[random.randint(0, len(self.classes) - 1)]])
                #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
                test_label.append(self.classes[random.randint(0, len(self.classes) - 1)])
            else:
                test_predictions.append([i, self.classes[test_probabilities[i]]])
                #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
                test_label.append(self.classes[test_probabilities[i]])
        generate_submission_csv(test_predictions, "naivebayes_sacdemots")

        #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
        return test_label


#Va etre tres similaire a NaiveBayesSacMots, par contre,  dans l'entrainement, on va differer
#le calcul des probabilié
class NaiveBayesLissage:
    def __init__(self, classes, alpha):
        self.classes = classes
        self.alpha = alpha

    def train(self, train_data):

        self.priorProbabilityEachClass = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            self.priorProbabilityEachClass[i] = train_data[1].count(self.classes[i])
        self.priorProbabilityEachClass = np.divide(self.priorProbabilityEachClass, len(train_data[1]))

        self.train_inputs = data_preprocessing(train_data[0])

        self.distributionArray = np.zeros(len(self.classes), dtype=np.object)
        self.num_words_in_bag = np.zeros(len(self.classes), dtype=np.object)

        for i in range(len(self.train_inputs)):
            j = np.where(self.classes == train_data[1][i])[0][0]
            if self.distributionArray[j] == 0:
                self.distributionArray[j] = Counter(self.train_inputs[i])
            else:
                self.distributionArray[j] += Counter(self.train_inputs[i])

        sum_words_class = np.zeros(len(self.classes))
        for i in range(len(self.classes)):
            sum_words_class[i] = sum(self.distributionArray[i].values())
        for i in range(len(self.classes)):
            self.distributionArray[i] = {key: value / sum_words_class[i] for key, value in self.distributionArray[i].items()}

    def compute_predictions(self, test_data):
        #Ceci reste comme avant
        preprocessed_test_data = data_preprocessing(test_data)
        prediction_array = np.ones((len(preprocessed_test_data), len(self.classes)))

        #Ignorer le runtime warning
        #RuntimeWarning: overflow encountered in reduce return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
        #https://stackoverflow.com/questions/49013049/runtimewarning-invalid-value-encountered-in-reduce
        #Celui ci nous explique quon passe probablement qqchose de NaN ou Inf dans la Matrice
        #Ca se peux aussi que non et que le programme prend juste un temps trop long pour sexecuter
        old_settings = np.seterr(all='ignore')

        for i in range(len(preprocessed_test_data)):
            for j in range(len(self.classes)):
                value = np.prod(np.vectorize(self.distributionArray[j].get)(preprocessed_test_data[i], 0))
                if(value != 0):
                    prediction_array[i][j] = value*self.priorProbabilityEachClass[j]
                else:
                    #nombre de valeur differente
                    #nb = 0
                    #for k in range(len(self.classes)):
                        #nb += len(self.distributionArray[k].keys())

                    #lissage
                    proba = []
                    for word in preprocessed_test_data[i]:
                        if word not in self.distributionArray[j].keys():
                            # alpha/ (nb de mot different dans la classe) + alpha*(nb de mot dans le sac)
                            temp = (self.alpha) / (len(self.distributionArray[j].keys()) + self.alpha*(len(preprocessed_test_data[i])))
                            proba.append(temp)

                    npproba = np.array(proba)
                    #On ignore les mot absent dans le calcul de la probabilite
                    value_ignorer = np.prod(np.vectorize(self.distributionArray[j].get)(preprocessed_test_data[i], 1))
                    value_absente = np.prod(npproba)

                    prediction_array[i][j] = value_ignorer*value_absente*self.priorProbabilityEachClass[j]

        test_probabilities = np.argmax(prediction_array, axis=1) #TODO verifier si pour valeur de argmax identique,

        test_predictions = []
        test_labels = []
        for i in range(len(preprocessed_test_data)):
            if test_probabilities[i] == 0:
                test_predictions.append([i, self.classes[random.randint(0, len(self.classes) - 1)]])
                #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
                test_labels.append(self.classes[random.randint(0, len(self.classes) - 1)])
            else:
                test_predictions.append([i, self.classes[test_probabilities[i]]])
                #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
                test_labels.append(self.classes[test_probabilities[i]])

        generate_submission_csv(test_predictions, "naivebayes_lissage_sacdemots")

        #TEMPORAIRE: SIMPLEMENT POUR LE TESTING
        return test_labels


def generate_submission_csv(data, classification_name):
    # Cette fonction genere deux fichiers .csv, identiques l'un a l'autre, d'une prediction pour la competition.
    # L'utilite de generer deux fichier est d'utiliser les fichiers de type _classification_name lors du test de
    # plusieurs methodes de classification et submission_latest pour la remise sur Kaggle (et ne pas se melanger)
    #
    # data est le tableau numpy des donnes
    # classification_name est le nom du modele de classification sous forme d'un string

    np.savetxt("data/submission_" + classification_name + ".csv", data, header=SUBMISSION_HEADER_CSV, delimiter=",",
               fmt='%s', comments='')
    np.savetxt("data/submission_latest.csv", data, header=SUBMISSION_HEADER_CSV, delimiter=",", fmt='%s', comments='')


def data_preprocessing(data):
    # Removes non alphabetic characters (tokenise rapide)
    # Makes words lower case
    # Remove stop words (the, a, we, etc)
    # Stemming (training -> train, running -> run) ?
    # TODO Remove spelling mistakes ?
    preprocessed_data = []
    existing_words = []
    ps = PorterStemmer()
    lem = WordNetLemmatizer()

    processing_range = data
    if IS_TEST:
        processing_range = data[0:1000]

    for i in range(len(processing_range)):
        tokenized_data = nltk.RegexpTokenizer(r'\w+').tokenize(data[i])  # TODO might want to split words like

        tokenized_data = [lem.lemmatize(token.lower(),"v")
                          for token in
                          tokenized_data if token.lower() not in STOP_WORDS_LIST]
        preprocessed_data.append(np.array(tokenized_data))

    return np.array(preprocessed_data)

def data_preprocessing_alternative(data):
    #https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
    preprocessed_data = []
    stop_words = set(stopwords.words("english"))
    stop_words.add('.')
    stop_words.add(',')
    stop_words.add('\'s')
    ps = PorterStemmer()
    lem = WordNetLemmatizer()


    for comment in data:
        # word tokenization
        tokenized_comment = word_tokenize(comment)

        #removing stopwords
        filtered_sent=[]

        for word in tokenized_comment:
            lematized_word = lem.lemmatize(word,"v")
            if lematized_word not in stop_words:
                filtered_sent.append(word)

        #adding the comment
        preprocessed_data.append(filtered_sent)

    return preprocessed_data


def precision_rate(train_inputs):
    #Slice
    train_set = (train_inputs[0][:60000],train_inputs[1][:60000])
    test_set = (train_inputs[0][60000:],train_inputs[1][60000:])

    # Generation liste de classes a predire.
    classes = []
    for i in train_set[1]:
        if i not in classes:
            classes.append(i)
    classes = np.array(classes)

    # 0.01 nous donne 0.4356
    # 0.0001  nous donne 0.4986
    # 0.00036 nous donne 0.4772
    # 0.00007 4986
    # 0.00005 499
    # 0.000026 5005
    # 0.00000021 0.501
    naiveBayesLissagePrediction = NaiveBayesLissage(classes,0.05)
    naiveBayesLissagePrediction.train(train_set)
    predicted_labels = naiveBayesLissagePrediction.compute_predictions(test_set[0])

    #calculate success rate
    goteem = 0
    real_labels = test_set[1]

    for i in range(len(real_labels)):
        if real_labels[i] == predicted_labels[i]:
            goteem += 1

    rate = goteem/len(real_labels)

    return rate

def main():
    # Generation d'un tuple (commentaire, subreddit)
    with open('data/data_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/data_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Generation liste de classes a predire.
    classes = []
    for i in train_data[1]:
        if i not in classes:
            classes.append(i)
    classes = np.array(classes)

    print("Starting at")

    print(datetime.datetime.now())


    #data_preprocessing(train_data)[0]
    #print(data_preprocessing_old(test_data)[0])
    #print(data_preprocessing(test_data)[0])

    #TEST POUR SCORE
    print(precision_rate(train_data))

    #print("Creating randomPrediction model")
    #randomPrediction = RandomPrediction(classes)
    #print("Creating naiveBayesSacMotsPrediction model")
    #naiveBayesSacMotsPrediction = NaiveBayesSacMots(classes)
    #print("Creating naiveBayesLissagePrediction model")
    #alpha decide le degré du lissage
    alpha = 0.0001
    #naiveBayesLissagePrediction = NaiveBayesLissage(classes,alpha)

    #print("Training randomPrediction model")
    #randomPrediction.train(train_data)
    #print("Training naiveBayesSacMotsPrediction model")
    #naiveBayesSacMotsPrediction.train(train_data)
    #print("Training naiveBayesLissagePrediction model")
    #naiveBayesLissagePrediction.train(train_data)

    #print(datetime.datetime.now())

    #print("Predicting randomPrediction model")
    #randomPrediction.compute_predictions(test_data)
    #print("Predicting naiveBayesSacMotsPrediction model")
    #naiveBayesSacMotsPrediction.compute_predictions(test_data)
    #print("Predicting naiveBayesLissagePrediction model")
    #naiveBayesLissagePrediction.compute_predictions(test_data)

    print("Done")
    print("Ended at")
    print(datetime.datetime.now())

main()
