import urllib2
#import urllib
from BeautifulSoup import *

def crawl(urls, retrieved_pages = {}, base = None):
    for url in [u.rstrip('/') for u in urls]:
        if url in retrieved_pages:
            continue
#        try:
        response = urllib2.urlopen(url)
 #       except HTTPError as e:
   #         print(e,url)
  #          continue
        page = parse(response,url,base)
        retrieved_pages[url] = page
        crawl(page[2], retrieved_pages,base)
    return sorted(retrieved_pages.values())

def parse(html,url,base):
    soup = BeautifulSoup(html, 'lxml')
    content = soup.body.get_text().strip()
    links = [urljoin(url,l.get('href')) for l in soup.findAll('a')]
    links = [l for l in links if urlparse(l).netloc in bases]
    return url,content, l

def show():
    urls = ['http://www.imdb.com/']
    pages = crawl(urls)
    for page in pages:
        print page
        print

show()


