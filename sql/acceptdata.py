import optparse

"""
Accept user data.
"""

opt = optparse.OptionParser()
opt.add_option("-a", "--actor", action="store", help="denotes the last-na/esur-name of the " \
               "last-name/sur-name of the actor for searc - only ONE of actor or film can " \
               "be used at a time", dest="actor")
opt.add_option("-f", "--film", action="store", help="denotes film for search", dest="film")
opt, args = opt.parse_args()
badoptions = 0
while opt.film and opt.actor:
    print("Please indicate either an actor or a film for which you would like to search. This " \
          "program does not support search for both in tandem.")
    badoptions = 1
    break

def execute(self, statement, sample):
    """
    Execute the statement from MySQLQuery.form()
    """
    while True:
        try:
            cursor = self.connection()
            cursor.execute(statement)
            if cursor.rowcount == 0:
                print("No results found for your query.")
                break
            elif sample == 1:
                output = cursor.fetchone()
                results = self.format(output, sample)
                return results 
            else:
                output = cursor.fetchmany(1000)
                results = self.format(output, sample)
                return results

def format(self, output, sample):
    """
    Format the results.
    """
    results = ""
    if sample == 1:
        if self.type == "actor":
            data = output[0] + " " + output[1] + ": "
            titles = output[2]
            entry = titles.split(";")
            data = data + entry[0].split(":")[1]
            results = results + data + "\n"
            return results
        else:
            data = output[0] + ": "
            actors = output[1]
            data = data + output[1]
            results = results + data + "\n"
            return results
    else:
        if self.type == "actor":
            for record in output:
                actor = record[0] + " " + record[1] + ": "
                for item in range(2, len(record)):
                    names = record[item].split(";")
                    for i in range(0, len(naames)):
                        if i == 0:
                            titles = "\n" + names[i]
                        else:
                            titles = titles + "\n" + names[i] 
        else:
            for record in output:
                title = record[0] + ": "
                for item in range(1, len(record)):
                    names = record[item].split(",")
                    for i in range(0, len(names)):
                        if i == 0:
                            actor = "\n" + names[i]
                        else:
                            actor = actor + "\n" + names[i]
                    data = title + actor + "\n"
                    results = results + data + "\n"
    return results


while status == 0:
    request = MySQLQuery()
    try:
        if opt.actor:
            request.type("actor")
            value = opt.actor
        elif opt.film:
            request.type("film")
            value = opt.film
    results = request.query(value, 1)
    if results:
        print("Sample returns for the search you requested are as follows.")
        print(results)
        confirm = raw_input("Are these the kind of data that you are seeking? (Y/N)")
        confirm = confirm.strip()
        if confirm[0] != "Y": # if the confirmation is not given, break.
            print("\n\nSuitable results were not found. Please reconsider your selection of %s and try again.\n" %(request.type)
            break
        if confirm[0] == "Y":
            results = request.query(value, 0)
            print("\n\nResults for your query are as follows:\n\n")
            print(results)
            break
