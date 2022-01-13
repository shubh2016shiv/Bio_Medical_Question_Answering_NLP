class TopicsByDiseasesOrGenetics:
    def __init__(self, ner_path, train_mode=False):
        self.ner_path = ner_path
        self.train_mode = False
        self.diseases = []
        self.genetics = []

    def readNERFile(self, ner_type: str):
        if ner_type == 'DISEASES':
            with open(self.ner_path + "/DiseasesNER.txt", encoding='utf-8') as file:
                ner_content = file.read().splitlines()
        elif ner_type == 'GENETICS':
            with open(self.ner_path + "/geneticsNER.txt", encoding='utf-8') as file:
                ner_content = file.read().splitlines()

        return ner_content

    def getDiseases(self):
        self.diseases = self.readNERFile(ner_type='DISEASES')
        return self.diseases

    def getGenetics(self):
        self.genetics = self.readNERFile(ner_type='GENETICS')
        return self.genetics
