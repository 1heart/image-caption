import cv2, os, numpy as np
from keras.preprocessing import text, sequence

class FlickrData:
    def __init__(self, path='data/Flickr8k.token.txt', n_vocab=100, max_seq_len=16):
        self.img_to_caps = dict()

        print('Opening file, creating img_to_caps...')
        with open(path, 'r') as f:
            for line in f:
                tokens = line.split(' ')
                img_fname, num = tokens[0].split('#')
                caption = ' '.join(tokens[1:]).strip()
                if img_fname not in self.img_to_caps: self.img_to_caps[img_fname] = []
                self.img_to_caps[img_fname].append(caption)
        self.img_fnames = list(self.img_to_caps.keys())

        print('Getting word counts, creating word_to_int and int_to_word...')
        texts = []
        self.tk = text.Tokenizer(nb_words=n_vocab)
        for img_fname in self.img_fnames:
            texts += self.img_to_caps[img_fname]

        self.tk.fit_on_texts(texts)
        self.sorted_word_counts = sorted(self.tk.word_counts.items(), key=lambda x: x[1])
        self.sorted_word_counts = self.sorted_word_counts[::-1][:n_vocab]
        self.word_to_int = {t[0]: i for i,t in enumerate(self.sorted_word_counts)}
        self.int_to_word = {i: t[0] for i,t in enumerate(self.sorted_word_counts)}
        
        print('Creating sequences...')
        img_to_seqs = {}
        
        for img_fname, captions in self.img_to_caps.items():
            seqs = []
            for caption in captions:
                seqs.append([self.word_to_int[w] for w in caption.split() if w in self.word_to_int])
            img_to_seqs[img_fname] = seqs
            
        print('Creating partial padded sequences and corresponding next chars...')
        self.img_to_padded_seqs, self.img_to_next_chars = {}, {}
        
        for img_fname, seqs in img_to_seqs.items():
            partial_seqs = []
            next_words = []
            for seq in seqs:
                for i in range(1,len(seq)):
                    partial_seqs.append(seq[:i])
                    next_words.append(seq[i])
            padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_seq_len)

            next_words_1hot = np.zeros([len(next_words), n_vocab], dtype=np.bool)
            for i,next_word in enumerate(next_words):
                next_words_1hot[i,next_word] = 1
            self.img_to_padded_seqs[img_fname] = padded_partial_seqs
            self.img_to_next_chars[img_fname] = next_words_1hot

                
    def get_img(self, img_fname, img_size=(299,299), path='data/Flicker8k_Dataset/flickr8k'):
        img_path = os.path.join(path, img_fname)
        return cv2.resize(cv2.imread(img_path), img_size).astype(np.float32)
