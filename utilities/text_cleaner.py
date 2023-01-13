import string
import re
import nltk
from nltk.stem import WordNetLemmatizer 


def advanced_data_cleaning(text):
    def replace_exclamations(x):
        """ Replaces multiple exclamation marks by the word exclamationMark """
        x = re.sub('(\! )+(?=(\!))', '', x)
        x = re.sub(r"(\!)+", ' exclamationMark ', x)
        return x

    def replace_points(x):
        """ Replaces multiple points by the word multiplePoints """
        x = re.sub('(\. )+(?=(\.))', '', x)
        x = re.sub(r"(\.)+", ' multistop ', x)
        return x


    def replace_questions(x):
        """ Replaces multiple question marks by the word questionMark """
        x = re.sub('(\? )+(?=(\?))', '', x)
        x = re.sub(r"(\?)+", ' questionMark ', x)
        return x

    def tokenization(text):
        text = re.split('\W+', text)
        return text

    def lemmatizer(l,text):
        text = [l.lemmatize(word) for word in text]
        return text

    def join_tokens(tokens):
        text = ' '.join(tokens)
        return text

    def split_negation(text):
        negations_dict = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                        "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                        "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                        "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                        "mustn't":"must not","ain't":"is not"}
        neg_pattern = re.compile(r'\b(' + '|'.join(negations_dict.keys()) + r')\b')
        text = neg_pattern.sub(lambda x: negations_dict[x.group()], text)
        return text

    def replace_contractions(text):
        contractions_dict = {"i'm":"i am", "wanna":"want to", "whi":"why", "gonna":"going to",
                        "wa":"was","nite":"night","there's":"there is","that's":"that is",
                        "ladi":"lady", "fav":"favorite", "becaus":"because","i\'ts":"it is",
                        "dammit":"damn it", "coz":"because", "ya":"you", "dunno": "do not know",
                        "donno":"do not know","donnow":"do not know","gimme":"give me"}
        contraction_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
        text = contraction_pattern.sub(lambda x: contractions_dict[x.group()], text)
        
        contraction_patterns = [(r'ew(\w+)', 'disgusting'),(r'argh(\w+)', 'argh'),(r'fack(\w+)', 'fuck'),
                                (r'sigh(\w+)', 'sigh'),(r'fuck(\w+)', 'fuck'),(r'omg(\w+)', 'omg'),
                                (r'oh my god(\w+)', 'omg'),(r'(\w+)n\'', '\g<1>ng'),(r'(\w+)n \'', '\g<1>ng'),
                                (r'(\w+)\'ll', '\g<1> will'),(r'(\w+)\'ve', '\g<1> have'),(r'(\w+)\'s', '\g<1> is'),
                                (r'(\w+)\'re', '\g<1> are'),(r'(\w+)\'d', '\g<1> would'),(r'&', 'and'),
                                ('y+a+y+', 'yay'),('y+[e,a]+s+', 'yes'),('n+o+', 'no'),('a+h+','ah'),('m+u+a+h+','kiss'),
                                (' y+u+p+ ', ' yes '),(' y+e+p+ ', ' yes '),(' idk ',' i do not know '),(' ima ', ' i am going to '),
                                (' nd ',' and '),(' dem ',' them '),(' n+a+h+ ', ' no '),(' n+a+ ', ' no '),(' w+o+w+', 'wow '),
                                (' w+o+a+ ', ' wow '),(' w+o+ ', ' wow '),(' a+w+ ', ' cute '), (' lmao ', ' haha '),(' gad ', ' god ')]
        patterns = [(re.compile(regex_exp, re.IGNORECASE), replacement)
                    for (regex_exp, replacement) in contraction_patterns]
        for (pattern, replacement) in patterns:
            (text, _) = re.subn(pattern, replacement, text)
        return text
        
    # Cleaning
    #Add trailing and leading whitespaces for the sake of preprocessing
    text = ' '+text+' '
    #lowercase tweet
    text = text.lower()
    #seperate negation words
    text = split_negation(text)
    # replace contradictions
    text = replace_contractions(text)
    #seperate punctuation from words
    text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text)
    #remove the observed pattern of numbers seen above
    text = re.sub(r'\\ [0-9]+ ', '', text)
    #replace ?,!,. by words
    text = replace_exclamations(text)
    text = replace_questions(text)
    text = replace_points(text)
    
    #Now since we translated punctuation and negative words we can remove the rest of the 'unwanted' chars
    #remove unwanted punctuation
    text = re.sub("[^a-zA-Z]", " ", text)
    
    #remove trailing and leading whitespace
    text = text.strip() 
    #remove multiple consecutive whitespaces
    text = re.sub(' +', ' ',text) 
    
    # Custom stopwords removal
    # stop_words = list(nltk.corpus.stopwords.words('english'))
    # safelist = ['aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    # blacklist = [word for word in stop_words if word not in safelist]


    #Lemmatization
    l = WordNetLemmatizer() 
    text = tokenization(text)
    # text = [word for word in text if word not in blacklist]
    text = join_tokens(lemmatizer(l,text))
    return text 