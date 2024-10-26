import nltk
import os
import pickle
import math
import re
import collections
import webbrowser
import tkinter as tk
from tkinter import *
import speech_recognition as sr
import requests
from nltk.stem.porter import PorterStemmer

# First, define all classes
class TrieNode:
    def __init__(self):
        self.val = None
        self.pointers = {}
        self.end = 0
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        self.recInsert(word, self.root)

    def recInsert(self, word, node):
        if len(word[:1]) == 0:
            node.end = 1
            return        
        if word[:1] not in node.pointers:
            newNode = TrieNode()
            newNode.val = word[:1]
            node.pointers[word[:1]] = newNode
            self.recInsert(word[1:], node)
        else:            
            nextNode = node.pointers[word[:1]]
            self.recInsert(word[1:], nextNode)

    def search(self, word):
        if len(word) == 0:
            return False
        return self.recSearch(word, self.root)

    def recSearch(self, word, node):
        if len(word[:1]) == 0:
            if node.end == 1:
                return True
            else:
                return False        
        elif word[:1] not in node.pointers:
            return False
        else:
            nextNode = node.pointers[word[:1]]
            return self.recSearch(word[1:], nextNode)

    def startsWith(self, prefix):
        if len(prefix) == 0:
            return True
        return self.recSearchPrefix(prefix, self.root)

    def recSearchPrefix(self, word, node):
        if len(word[:1]) == 0:
            return True        
        elif word[:1] not in node.pointers:
            return False
        else:
            nextNode = node.pointers[word[:1]]
            return self.recSearchPrefix(word[1:], nextNode)

    def findAll(self, node, word, sugg):
        lis = list(map(chr, range(97, 123)))
        lis.append("'")
        for c in lis:
            if c in node.pointers:
                if node.pointers[c].end == 1:
                    sugg.append(word + str(c))
                self.findAll(node.pointers[c], word + str(c), sugg)
        return

    def didUMean(self, word, sugg):
        if self.startsWith(word):
            top = self.root
            for c in word:
                top = top.pointers[c]
            self.findAll(top, word, sugg)
        else:
            return

# Create global trie instance
trie = Trie()

# Define helper functions
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        if model[f] > 1 or trie.search(f):
            model[f] += 1
    return model

def get_words(text): 
    return re.findall('[a-z]+', text.lower())

# Load data and initialize global variables
words_list = pickle.load(open("dist_words.p", "rb"))
for word in words_list:
    trie.insert(word.lower())

NWORDS = train(get_words(open('big.txt', 'r').read()))

class EditDist:
    def __init__(self):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def edits1(self, word):
       splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
       deletes = [a + b[1:] for a, b in splits if b]
       transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
       replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
       inserts = [a + c + b for a, b in splits for c in self.alphabet]
       return set(deletes + transposes + replaces + inserts)

    def knownEdits2(self, word):
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if trie.search(e2))

    def known(self, words):
        return set(w for w in words if w in NWORDS)

    def correct(self, word):
        candidates = self.known([word]) or self.known(self.edits1(word)) or self.knownEdits2(word) or [word]
        sugg = list(candidates)
        sugg.sort(key=lambda s: nltk.edit_distance(word, s))
        return sugg[:min(len(sugg), 10)]

# [Rest of the classes remain the same: IntegratedSearchApp and AutocompleteEntry]

class AutocompleteEntry(Entry):
    def __init__(self, *args, **kwargs):
        Entry.__init__(self, *args, **kwargs)
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)
        self.lb_up = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            if self.lb_up:
                self.lb.destroy()
                self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = Listbox()
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.place(x=self.winfo_x()+self.winfo_width()-35, y=self.winfo_y()+self.winfo_height())
                    self.lb_up = True

                self.lb.delete(0, END)
                for w in words:
                    self.lb.insert(END, w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False

    def selection(self, event):
        if self.lb_up:
            self.var.set(self.lb.get(ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(END)

    def up(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index)-1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def down(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != END:
                self.lb.selection_clear(first=index)
                index = str(int(index)+1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        word = self.var.get().lower()
        ed = EditDist()
        sugg = []
        trie.didUMean(word, sugg)
        
        if len(sugg) != 0:
            sugg.sort(key=lambda s: len(s))
        else:
            sugg = ed.correct(word)
            
        if trie.search(word):
            sugg.insert(0, word)
        
        return sugg[:min(len(sugg), 10)]

class IntegratedSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Search")
        self.root.geometry('800x600')

        # Initialize data structures
        self.trie = trie  # Use global trie
        self.words = words_list  # Use global words_list
        self.dict = pickle.load(open("termFr_dict.p", "rb"))
        self.invertedIndex = pickle.load(open("invertedIndex.p", "rb"))
        self.termFr_idf = pickle.load(open("termFr_idf.p", "rb"))
        self.lengths = {}
        self.stemmer = PorterStemmer()
        self.N = len(self.dict)

        # Calculate lengths for TF-IDF
        for key in self.termFr_idf:
            temp = 0.0
            for word in self.termFr_idf[key]:
                temp += self.termFr_idf[key][word] * self.termFr_idf[key][word]
            self.lengths[key] = math.sqrt(temp)

        # Create GUI elements
        self.create_gui()

    def create_gui(self):
        # Search frame
        frame1 = Frame(self.root)
        frame1.pack(pady=10)

        self.entry = AutocompleteEntry(frame1, width=40)
        self.entry.pack(side=LEFT, padx=5)
        
        search_button = Button(frame1, text='Search', width=15, command=self.show_search_results)
        search_button.pack(side=LEFT, padx=5)
        
        speak_button = Button(frame1, text='Speak Now', width=10, command=self.speak_now)
        speak_button.pack(side=LEFT, padx=5)

        # Results frame
        results_frame = Frame(self.root)
        results_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # Word meaning section
        meaning_frame = LabelFrame(results_frame, text="Word Definition", padx=5, pady=5)
        meaning_frame.pack(fill='x', pady=5)
        
        self.meaning_text = Text(meaning_frame, wrap='word', height=4)
        self.meaning_text.pack(fill='x')

        # Document results section
        docs_frame = LabelFrame(results_frame, text="Related Documents", padx=5, pady=5)
        docs_frame.pack(fill='both', expand=True)
        
        self.content_text = Text(docs_frame, wrap='word')
        self.content_text.pack(expand=True, fill='both', side='left')
        
        scroll_bar = Scrollbar(docs_frame)
        scroll_bar.pack(side='right', fill='y')
        
        self.content_text.configure(yscrollcommand=scroll_bar.set)
        scroll_bar.config(command=self.content_text.yview)

    def fetch_meaning(self, word):
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                meaning = data[0]["meanings"][0]["definitions"][0]["definition"]
                self.meaning_text.delete('1.0', END)
                self.meaning_text.insert('1.0', f"{word}: {meaning}")
            else:
                self.meaning_text.delete('1.0', END)
                self.meaning_text.insert('1.0', f"Definition not found for '{word}'")
        except Exception as e:
            self.meaning_text.delete('1.0', END)
            self.meaning_text.insert('1.0', f"Error fetching definition: {str(e)}")

    def page_rank(self, query):
        query_dic = {}
        q_list = []
        
        for word in query.split():
            word = word.lower()
            word = self.stemmer.stem(word)
            query_dic[word] = query_dic.get(word, 0) + 1

        for key in query_dic:
            q_list.append(key)

        score = {}

        for word in q_list:
            if word in self.invertedIndex:
                df = len(self.invertedIndex[word])
                idf = math.log(self.N / (df * 1.0), 10.0)
                wtq = idf * (1.0 + math.log(query_dic[word], 10.0))
                
                for doc in self.invertedIndex[word]:
                    temp = score.get(doc, 0)
                    wtd = self.termFr_idf[doc][word]
                    score[doc] = temp + wtq * wtd

        ranking = []

        for key in score:
            score[key] = score[key] / self.lengths[key]
            ranking.append((key, score[key]))

        ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
        return ranking[:min(len(ranking), 20)]

    def show_search_results(self):
        query = self.entry.get()
        
        # Get word meaning
        self.fetch_meaning(query.lower())
        
        # Get document results
        self.content_text.delete('1.0', END)
        ranking = self.page_rank(query)
        
        if ranking:
            for doc, score in ranking:
                label = tk.Label(self.content_text, text=f"brown/{doc}", fg="blue", cursor="hand2")
                label.bind("<Button-1>", self.on_doc_click)
                self.content_text.window_create('end', window=label)
                self.content_text.insert('end', '\n')
        else:
            self.content_text.insert('1.0', "No relevant documents found.")

    def on_doc_click(self, event):
        doc_path = event.widget.cget("text")
        webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open(doc_path)

    def speak_now(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("Recognized:", text)
            self.entry.delete(0, END)
            self.entry.insert(0, text)
            self.show_search_results()
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results: {e}")

class AutocompleteEntry(Entry):
    def __init__(self, *args, **kwargs):
        Entry.__init__(self, *args, **kwargs)
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = StringVar()

        self.var.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)
        self.lb_up = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            if self.lb_up:
                self.lb.destroy()
                self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = Listbox()
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Right>", self.selection)
                    self.lb.place(x=self.winfo_x()+self.winfo_width()-35, y=self.winfo_y()+self.winfo_height())
                    self.lb_up = True

                self.lb.delete(0, END)
                for w in words:
                    self.lb.insert(END, w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False

    def selection(self, event):
        if self.lb_up:
            self.var.set(self.lb.get(ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(END)

    def up(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index)-1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def down(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != END:
                self.lb.selection_clear(first=index)
                index = str(int(index)+1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        word = self.var.get().lower()
        ed = EditDist()
        sugg = []
        trie.didUMean(word, sugg)
        
        if len(sugg) != 0:
            sugg.sort(key=lambda s: len(s))
        else:
            sugg = ed.correct(word)
            
        if trie.search(word):
            sugg.insert(0, word)
        
        return sugg[:min(len(sugg), 10)]

if __name__ == "__main__":
    # Initialize global variables
    trie = Trie()
    
    # Load word list
    words_list = pickle.load(open("dist_words.p", "rb"))
    for word in words_list:
        trie.insert(word.lower())
        
    # Create NWORDS - fixed version
    try:
        with open('big.txt', 'r') as file:
            text = file.read()
            NWORDS = train(get_words(text))
    except FileNotFoundError:
        print("Error: big.txt not found")
        NWORDS = collections.defaultdict(lambda: 1)

    # Start the application
    root = Tk()
    app = IntegratedSearchApp(root)
    root.mainloop()