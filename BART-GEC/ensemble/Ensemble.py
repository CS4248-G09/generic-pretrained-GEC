import glob
import numpy as np
import sys
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class Loader():

    OUTPUT_PATH = "Outputs/*.txt"
    BEST_PATH = "Outputs/Best/*.txt"
    INPUT_PATH = "Input/*.txt"

    def __init__(self):
        all_output_files = self.get_output_file_names()
        self.outputs = np.array([self.read_text_file(fn) for fn in all_output_files])
        self.best = self.read_text_file(self.get_best_name())
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.input = self.read_text_file(self.get_input_name())
        return

    def read_text_file(self,file_path):
        file = []
        with open(file_path, 'r') as f:
            file = f.readlines()
        return file

    def get_output_file_names(self):
        return glob.glob(self.OUTPUT_PATH)

    def get_best_name(self):
        return glob.glob(self.BEST_PATH)[0]

    def get_input_name(self):
        return glob.glob(self.INPUT_PATH)[0]

    def get_line(self, index):
        return self.outputs[:,index]

    def get_input_line(self, index):
        return self.input[index]

    def get_best_line(self, index):
        return self.best[index]

    def get_line_similarity(self, index):
        outputs_line = self.get_line(index)
        n = self.get_num_outputs()
        res = np.zeros(n)
        sentence_embeddings = self.model.encode(outputs_line)
        for i in range(n): 
            similarity = cosine_similarity([sentence_embeddings[i]], sentence_embeddings)
            res[i] = np.average(similarity)
        
        return res

    def get_num_lines(self):
        return len(self.outputs[0])
    
    def get_num_outputs(self):
        return len(self.outputs)


class Ensember():

    RANDOM = 0
    TOKEN_VOTE = 1
    COSINE = 2
    TOKEN_COMBINE = 3
    MAX_VOTING = 4
    ASSSUMPTION_MAX_VOTING = 5
    ALL = -1

    PATH = "EnsembleOutput/Output"

    def __init__(self, flag):
        self.loader = Loader()
        self.FLAG = int(flag)
        return

    def get_all(self):
        for i in range(6):
            self.get_ensemble_output(i)

    def get_output(self):
        self.get_ensemble_output(self.FLAG)
    
    def get_ensemble_output(self, flag):
        
        ensember_output = ""
        name = ""

        if (flag == self.ALL):
            return self.get_all()

        if (flag == self.RANDOM):
            ensember_output = self.random_ensemble()
            name = "_random.txt"
        if (flag == self.TOKEN_VOTE):
            ensember_output = self.token_vote_ensemble()
            name = "_max_voting_token.txt"
        if (flag == self.COSINE):
            ensember_output = self.cosine_ensemble()
            name = "_cosine.txt"
        if (flag == self.MAX_VOTING):
            ensember_output = self.max_voting_ensemble()
            name = "_max_voting.txt"

        if (flag == self.ASSSUMPTION_MAX_VOTING):
            ensember_output = self.max_vote_assumption_ensemble()
            name = "_max_voting_assumption.txt"

        
        self.save_output(ensember_output, name)


    def save_output(self, output, name):
        file = open(self.PATH + name, 'w')
        file.write(output)

    def get_special_sentence(self, sentence):
        res = []
        tokens = {}
        s = "$"
        for token in sentence.split(" "):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 0
            
            c = tokens[token] * s
            res.append(token + c)

        return res

    def token_vote_ensemble(self):
        
        l = self.loader.get_num_lines()
        n = self.loader.get_num_outputs()
        token_vote_ensemble_output = []

        for i in range(l):
            sentences = [self.get_special_sentence(s) for s in self.loader.get_line(i)]
            token_votes = {}
            for j in range(n):
                for token in sentences[j]:
                    if token in token_votes:
                        token_votes[token] += 1
                    else:
                        token_votes[token] = 1

            
            for token in token_votes:
                token_votes[token] = token_votes[token]*token_votes[token]

            scores = np.zeros(n)
            for j in range(n):
                score = 0
                for token in sentences[j]:
                    score += token_votes[token]
                scores[j] = score
            index = np.argmax(scores)

            token_vote_ensemble_output.append(self.loader.get_line(i)[index])
        
        return "".join(line for line in token_vote_ensemble_output)


    def cosine_ensemble(self):
        l = self.loader.get_num_lines()
        
        cosine_ensemble_output = []

        for i in range(l):
            sentences = self.loader.get_line(i)
            similarities = self.loader.get_line_similarity(i)
            index = np.argmax(similarities)
            cosine_ensemble_output.append(sentences[index])

        return "".join(line for line in cosine_ensemble_output)

    def get_max_vote(self, ind, s=[]):
        n = self.loader.get_num_outputs()
        sentences = self.loader.get_line(ind)
        if len(s) != 0:
            sentences = s
            n = len(s)
        voting = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if sentences[i] == sentences[j]:
                    voting[i]+=1
        index = np.argmax(voting)
        res = sentences[index]
        if np.all(voting == voting[0]):
            if len(s) == 0:
                res = self.loader.get_best_line(ind)
            else:
                res = s[random.randint(0, len(s)-1)]
        
        return res

    def max_voting_ensemble(self):
        l = self.loader.get_num_lines()
        voting_output = []
        for i in range(l):
            toAdd = self.get_max_vote(i)
            voting_output.append(toAdd)
        return "".join(line for line in voting_output)

    def max_vote_assumption_ensemble(self):
        l = self.loader.get_num_lines() 
        assumption_voting_output = []
        for i in range(l):
            list_line = self.loader.get_line(i)
            input_line = self.loader.get_input_line(i)
            toAdd = input_line
            counter = 0
            for s in list_line:
                if s == input_line:
                    counter = counter + 1
            
            if counter != self.loader.get_num_outputs():
                newLine = []
                for line in list_line:
                    if line != input_line:
                        newLine.append(line)
                toAdd = self.get_max_vote(-1,newLine)
            assumption_voting_output.append(toAdd)

        return "".join(line for line in assumption_voting_output)

    def random_ensemble(self):

        l = self.loader.get_num_lines()
        n = self.loader.get_num_outputs()
        random_ensemble_output = []

        for i in range(l):
            sentences = self.loader.get_line(i)
            random_ensemble_output.append(sentences[random.randint(0,n-1)])
        return "".join(line for line in random_ensemble_output)

if __name__ == "__main__":
    ensember = Ensember(sys.argv[1])
    ensember.get_output()




