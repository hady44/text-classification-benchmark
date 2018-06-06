import rdflib , pprint
from rdflib import URIRef, Graph
from rdflib.plugins import sparql


graph = Graph("Sleepycat")
graph.open("store", create=True)
graph.parse("/media/hady/My Passport/diego/yego/yago3.1_entire_ttl/yagoConteXtFacts_en.ttl", format="n3")

i = 0
for subject, predicate, object in graph:
    print subject, predicate, object
    i+=1
    if i > 100:
        break