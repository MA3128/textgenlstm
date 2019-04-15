class text_gen():
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import CuDNNLSTM, Bidirectional, Dense, Dropout

    def __init__(self):
        self.a = None
        
    def make_model(self, X_modified, Y_modified, from_file=False):
        from keras.models import Sequential
        from keras.layers import CuDNNLSTM, Bidirectional, Dense, Dropout, InputLayer
        model = Sequential()
        model.add(InputLayer(input_shape=(X_modified.shape[1], X_modified.shape[2])))
        model.add(CuDNNLSTM(800, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(700, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNLSTM(700))
        model.add(Dropout(0.2))
        if from_file:
            model.add(Dense(Y_modified.shape[1], activation='softmax'))
        else:
            model.add(Dense(52, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
    
    def prep_data(self, file, length=130):
        
        
        return X_modified, Y_modified, text_in_words, X, n_to_char, char_to_n
    
    def train_generate_from_file(self ,X_modified, Y_modified, from_file, batch_size=100, epochs=25):
        model = self.make_model(X_modified, Y_modified, from_file)
        model.fit(X_modified, Y_modified, epochs=epochs, batch_size=batch_size)
        model.save_weights('news_generatorchar.h5')
        return model
    
    def generate_text(self, file='headlines.txt', from_file=False, batch_size=100, epochs=25, nsamples=10, length=130):
        from keras.utils import np_utils
        import numpy as np
        import pandas as pd
        import io
        import numpy as np
        from keras.utils import np_utils
        corpus = file
        with io.open(corpus,encoding='ISO-8859-1') as f:
                text = f.read().lower().replace('\n', ' \n ')
        text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']

        text = (open(corpus, encoding='ISO-8859-1').read())
        text=text.lower()
        characters = sorted(list(set(text)))

        n_to_char = {n:char for n, char in enumerate(characters)}
        char_to_n = {char:n for n, char in enumerate(characters)}

        X = []
        Y = []
        length = len(text)
        seq_length = 100
        for i in range(0, length-seq_length, 1):
            sequence = text[i:i + seq_length]
            label =text[i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y.append(char_to_n[label])

        X_modified = np.reshape(X, (len(X), seq_length, 1))
        X_modified = X_modified / float(len(characters))
        Y_modified = np_utils.to_categorical(Y)
        
        if from_file:
            model = self.train_generate_from_file(X_modified, Y_modified,
                                                  batch_size, epochs, from_file)
        else:
            model = self.make_model(X_modified, Y_modified)
            model.load_weights('news_generatorchar.h5')
        
        
        news = []
        for i in range(nsamples):
            index = np.random.randint(len(X))
            string_mapped = X[index]
            if len(string_mapped)>100:
                continue
            full_string = []
            full_string = [n_to_char[value] for value in string_mapped]
            s = len(string_mapped) - 100 
            # generating characters
            for i in range(5):
                x = np.reshape(string_mapped, (1,len(string_mapped) - s, 1))
                x = x/float(len(characters))

                pred_index = np.argmax(model.predict(x, verbose=0))
                seq = [n_to_char[value] for value in string_mapped]
                full_string.append(n_to_char[pred_index])

                string_mapped.append(pred_index)
                string_mapped = string_mapped[1:len(string_mapped)]
            #combining text
            txt=""
            for char in full_string:
                txt = txt+char
            news.append(txt)
            
        clean_news = []
        for n in news:
            n = [w for w in n.split() if w in text_in_words]
            clean_news.append(' '.join(n))
        
        generated_news = pd.DataFrame(clean_news, columns=['News'])
    
        return generated_news