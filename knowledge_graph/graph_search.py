import ssl
import certifi
from neo4j import GraphDatabase
# from semantic_search import semantic_search

ssl_context = ssl.create_default_context(cafile=certifi.where())
uri = "neo4j://d651c50c.databases.neo4j.io:7687" 
user = "neo4j"
password = "63UdVpdlTaul-doTU481pOmYYnU91ekyzvFBqjbBH44"

driver = GraphDatabase.driver(
    uri,
    auth=(user, password),
    ssl_context=ssl_context,
    encrypted=True 
)
# Sample check for graph working
# with driver.session() as session:
#     print(session.run("RETURN 1 AS ok").single()["ok"])

def get_kural_details(kural_numbers):
    query = """
    MATCH (k:Kural)
    WHERE k.id IN $kural_numbers
    RETURN k { .* } AS kural_details
    """
    with driver.session() as session:
        result = session.run(query, kural_numbers=kural_numbers)
        data = [record["kural_details"] for record in result]
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)

# Sample working
# query = "what about betrayal"
# res = semantic_search(query)
# r = []
# for i in res:
#     r.append(i['number'])
# # print(r)
    
# # kural_nums = [1197, 137, 1094]
# print(get_kural_details(r))

