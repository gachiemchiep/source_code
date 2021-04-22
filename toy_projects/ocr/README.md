
# Tutorial to train scene text detection + recognition for Japanese


## Text detection 

### Generating Synthetic text detection dataset

* nltk

```bash

pip install nltk

# download nltk library 
import nltk
nltk.download()

# training text are take from tesseract training data
https://github.com/tesseract-ocr/langdata/tree/master/jpn
# jpn.training_text 
# words are already splitted by space

# we can use nltk to create "splited by space text"
# http://www.nltk.org/book-jp/ch12.html

```

* prepare the fonts 

```bash
sudo apt install fonts-ipa* fonts-mona fonts-takao* fonts-vlgothic 
```

* generate SynthText 

```bash
#
data
├── dset.h5
├── fonts
│   ├── fontlist.txt                        : your font list
│   ├── ubuntu
│   ├── ubuntucondensed
│   ├── ubuntujapanese                      : your japanese font
│   └── ubuntumono
├── models
│   ├── char_freq.cp
│   ├── colors_new.cp
│   └── font_px2pt.cp
└── newsgroup
    └── newsgroup.txt                       : your text source

```

## Text recognition



#


* SynthText 

```bash
# Don't use the python3 branch it's buggy

```


