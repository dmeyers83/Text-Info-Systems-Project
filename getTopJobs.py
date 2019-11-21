from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import re
import urllib
import pickle
import time

def main():
    options = Options()
    options.headless = True
    browser = webdriver.Chrome('./chromedriver',options=options)
    top50BIjobs = getBIJobs('https://www.businessinsider.com/best-jobs-in-america-2019-1',browser)
    top100USjobs = getUSNewsJobs('https://money.usnews.com/careers/best-jobs/rankings/the-100-best-jobs',browser)
    mergeListsandWrite(top50BIjobs,top100USjobs)
    browser.close()
    browser.quit()


def mergeListsandWrite(top50BIjobs,top100USjobs):
    phandle = open("topjobs.bin","rb")
    data=pickle.load(phandle)
    phandle.close()
    data=[]
    for i in top50BIjobs:
        if i in data:
            pass
        else:
            data.append(i)

    for j in top100USjobs:
        if j in data:
            pass
        else:
            data.append(j)
    phandle = open("topjobs.bin","wb")
    pickle.dump(data, phandle)
    phandle.close()

def get_js_soup(url,browser):
    browser.get(url)
    res_html = browser.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser')
    return soup

def get_annoyingUSN(url,browser):
    browser.get(url)
    elem = browser.find_element_by_tag_name("body")

    no_of_pagedowns = 20

    while no_of_pagedowns:
        elem.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        no_of_pagedowns-=1
        
    res_html = browser.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser')
    return soup

def getBIJobs(dir_url,browser):
    print ('-'*20,'Scraping Business Insider','-'*20)
    soup = get_js_soup(dir_url,browser)
    data=soup.find_all('h2',class_='slide-title-text')
    jobs=[]
    for row in data:
        job=row.text
        job=job[job.find(".")+2:]
        jobs.append(job)
    return jobs

def getUSNewsJobs(dir_url,browser):
    print ('-'*20,'Scraping USNews','-'*20)
    soup = get_annoyingUSN(dir_url,browser)
    data=soup.find_all('h2',class_='sc-bdVaJa hSILDx')
    jobs=[]
    for row in data:
        job=row.text
        jobs.append(job)
    return jobs

if __name__== "__main__":
  main()
