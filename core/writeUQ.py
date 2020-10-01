#####################################################
#    Tools for printing or writing data in file
#####################################################
# Saleh Rezaeiravesh,   salehr@kth.se
#----------------------------------------------------
#
def printRepeated(string_to_expand, length):
    """ 
    Repeats the string `string_to_extend`, `length` times 
    """
    return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]
#
