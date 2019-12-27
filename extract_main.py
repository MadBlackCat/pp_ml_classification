import os
import re
import shutil
from bs4 import BeautifulSoup
from bs4 import Comment
from readability import Document


def html_to_text(html_file):

    soup = BeautifulSoup(html_file, features="lxml")
    # print(soup.get_text())
    [x.extract() for x in soup.find_all('script')]
    [x.extract() for x in soup.find_all('style')]
    [x.extract() for x in soup.find_all('meta')]
    [x.extract() for x in soup.find_all('noscript')]
    [x.extract() for x in soup.find_all('a')]
    [x.extract() for x in soup.find_all(text=lambda text: isinstance(text, Comment))]
    segments = soup.get_text
    text = '\n'.join(segments)
    pure_text = re.sub('\s+', ' ', text)
    return pure_text


def detect_main_text(html_file):
    # 查看哪种文件读取不出来
    try:
        with open(html_file, "rb") as f:
            doc = Document(f.read())
            return doc.summary()
    except:
        print(html_file)


def extract_title(html_file):
    soup = BeautifulSoup(html_file, features="lxml")
    maybe_title_tag = ['strong', 'b', 'em', 'i']
    title = []
    for tag in maybe_title_tag:
        may_title = soup.find_all(tag)

        title.append(i for i in may_title if len(i.strip()) > 0)

    return len(title)


path = "./validation_html/"
pp_dir = os.listdir(path)
for pp in pp_dir:
    clean_html = detect_main_text(path + pp)

    with open("./validation_main_text/"+pp, "w", encoding="utf-8") as f:
        f.write(clean_html)




