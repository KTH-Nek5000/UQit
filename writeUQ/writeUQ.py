#####################################################
#    Tools for writing data in file
#####################################################
# Saleh Rezaeiravesh,   salehr@kth.se
#----------------------------------------------------

def printRepeated(string_to_expand, length):
    """ Repeat string_to_extend up to a certain length """
    return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]



