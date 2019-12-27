"""
 　　#!/usr/bin/python3
    Author: RjMonkey  
    Date: 2019/11/3

    隐私政策文件的乱七八糟的整理
    有将ul，li找到并赋予start的代码

"""
import os
import shutil
import time
from selenium import webdriver
from bs4 import BeautifulSoup


def extract167():
    # all_txt = [i.rstrip(".txt") for i in os.listdir("./all_txt_file/")]
    # all_html = [i.rstrip(".html") for i in os.listdir("./all_html_file/")]
    file_167 = os.listdir("./labeled_set/")
    for i in file_167:
        try:
            shutil.copy("./all_html_file/" + i.replace(".txt.ann", ".html").replace("grj.", ""), "./167_html/")
        except: print(i.replace(".txt.ann", ".html").replace("grj.", ""))
        try:
            shutil.copy("./all_txt_file/" + i.replace(".txt.ann", ".txt").replace("grj.", ""), "./167_txt/")
        except: print(i.replace(".txt.ann", ".txt").replace("grj.", ""))


def replace_ul_tag(html):
    soup = BeautifulSoup(open(html, "rb"), features="lxml")
    li_tag = soup.find_all("li")
    ul_tag = soup.find_all("ul")
    for tag in li_tag:
        # new_tag = soup.new_tag
        new_string = " ".join(["[@Start#]" + i for i in tag.stripped_strings])

        tag.replace_with(new_string)
        # tag.unwrap()
    for tag in ul_tag:
        # tag.string = "[@Start#]"
        tag.unwrap()
    # # soup.li.unwrap()
    return soup


def first_merge_to_html(orgin_path, out_path):
    html_dir = os.listdir(orgin_path)
    for pp in html_dir:
        pp_html = replace_ul_tag(orgin_path+pp)
        with open(out_path+pp, "w", encoding="utf8") as f:
            f.write(str(pp_html))


def html_to_txt(path):

    html_dir = os.listdir(path)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--lang=en-US')
    # chrome_options.add_argument('--proxy-server=http://174.138.46.194:8080')
    prefs = {"profile.managed_default_content_settings.images": 2, "int1.accept_language": "en-GB"}

    chrome_options.add_experimental_option('prefs', prefs)
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.set_page_load_timeout(180)

    for pp in html_dir:
        try:
            driver.get(path + pp)
            html_text = driver.find_elements_by_xpath("/；；‘。’/body")[0].text
            html = driver.page_source
            time.sleep(3)
            with open('./validation_txt/' + pp + '.txt', 'w', encoding="utf-8") as f:
                f.write(str(html_text))
        except: print(pp)


def check_dir_diff(path1, path2):
    suffix1 = path1.split('.')[-1]
    suffix2 = path2.split('.')[-1]


# first_merge_to_html(orgin_path="./validation_html/", out_path="./validation_merged_html/")
html_to_txt("./validation_merged_html/")
