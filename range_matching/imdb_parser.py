import scrapy
from scrapy.http import HtmlResponse
from scrapy import Selector
import requests
import json
import pickle
import os

import logging
logging.getLogger('scrapy').disabled=True
logging.getLogger().setLevel(logging.WARNING)

class ActorItem(scrapy.Item):
    name = scrapy.Field()
    born = scrapy.Field()
    movies = scrapy.Field()
    url = scrapy.Field()
    bio = scrapy.Field()

class MovieItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    cast = scrapy.Field()

IS_ACTOR = False
    
class ImdbSpider(scrapy.Spider):
    name = 'imdb'
    allowed_domains = ["imdb.com"]
    base_url = "https://www.imdb.com"
    start_urls = ["https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m"]
    
    def parse(self, response):
        url_parts = response.xpath(".//*[@class='lister-item-content']/h3[@class='lister-item-header']"
                                   "/a/@href").extract()
        actor_url_list = [self.base_url + x +'/' for x in url_parts]
        if IS_ACTOR:
            for actor_url in actor_url_list:
                yield scrapy.Request(actor_url,
                                     callback=self.parse_actor,
                                     meta={'url':actor_url})
        else:
            if not os.path.exists('gen_movie.pkl'):
                movie_url_list = list(self.gen_movie(actor_url_list))
                with open('gen_movie.pkl', 'wb') as f:
                    pickle.dump(movie_url_list, f)
            else:
                with opne('gen_movie.pkl', 'rb') as f:
                    movie_url_list = pickle.load(f)
            
            for movie_url in movie_url_list:
                yield scrapy.Request(movie_url + 'fullcredits',
                                     callback=self.parse_cast,
                                     meta={'url': movie_url})
    
    def gen_movie(self, actor_url_list):
        already_parsed = set()
        for actor_url in actor_url_list:
            response = HtmlResponse(url=actor_url, body=requests.get(actor_url).content)
            movie_url_list = Selector(response=response).xpath(".//div[@class='filmo-category-section']"
                                                               "/div/b/a/@href").extract()
            movie_url_list = set(movie_url_list) - already_parsed
            for movie_url in movie_url_list:
                movie_url = self.base_url + movie_url
                yield movie_url
            already_parsed |= movie_url_list
       
    def parse_cast(self, response):
        item = MovieItem()
        
        item['url'] = response.meta['url']
        item['title'] = response.xpath(".//div[@class='subpage_title_block__right-column']"
                    "/div[@class='parent']/h3/a/text()").extract_first().strip()
        res = response.xpath(".//table[@class='cast_list']//tr/td[2]/a/text()").extract()
        item['cast'] = list(map(str.strip, res))
        return item
        
    def parse_actor(self, response):
        item = ActorItem()
        
        item['name'] = response.xpath(".//td[@class='name-overview-widget__section']"
                                     "/h1/span/text()").extract_first().strip()
        
        bio_part = response.xpath(".//div[@class='name-trivia-bio-text']/div/"
                    "/descendant-or-self::text()").extract()
        stop_word ="See full bio"
        bio_str_list = []
        for r in bio_part:
            if r == stop_word:
                break
            bio_str_list.append(r)
        item['bio'] = " ".join(bio_str_list).strip()
        
        item['born'] = response.xpath(".//time/@datetime").extract_first()
        
        item['url'] = response.meta['url']
        
        movie_list = response.xpath(".//div[@class='filmo-category-section']/div/b/a/text()").extract()
        item['movies'] = list(map(str.strip, movie_list[:15]))

        return item

from scrapy.crawler import CrawlerProcess

process = CrawlerProcess(
    settings={
        "FEEDS": {
            "actors.json" if IS_ACTOR else "movies.json": {"format": "json"}
        }
    }
)
process.crawl(ImdbSpider)
process.start()