############################################################
# Construct databases for the channel flow simulation data
############################################################
# Saleh Rezaeiravesh,  salehr@kth.se
#-----------------------------------------------------------
import nekProfsReader_tChan
#
def dbMakerCase_multiPar(prePath,caseName,interpOpts):
    """ 
    Make a sorted list of databases for a channel flow simulations case <caseName> which have 2 or more uncertain parameters and are simulated by either Nek5000 or OpenFOAM.   
    NOTE: A case contains several channel flow simulation.
    NOTE: for OpenFOAM, interpOpts={}
    -The database of each simulation included in the case contains statsitics profiles of channel flow.   
    -Each channel flow simulation corresponds a unique combination of uncertain parameters. We assume tensor-product was considered when designing the simulations case. 
    -Sorting is required to make sure our convention for tensor-product parameter samples is in place. (convention: the latest parameter should be in the outermost loop)
    -First the db of statsitics profiles of all simulations included in a case are read. A list of these db's is created. Then the information about the simulations included in the case (name and associated parameter samples) are read and sorted according to the convention. Finally, the list of db's is accordingly sorted and the entries of the information-db is merged with that. 
    """
    #(a) Read information about the case. This db is sorted according to our convention for tensor product (last parameter has the outermost loop)
    db_info=nekProfsReader_tChan.sortedCaseInfo(prePath,caseName)
    nSamples=db_info['nSamples']   #list of number of samples 
    nSamplesProd=1
    for i in range(len(nSamples)):
        nSamplesProd*=nSamples[i]
    #(b) Read in data in the order that oringinally ran (in Excel sheet)
    db=[];  #list of databases of profiles
    for i in range(nSamplesProd):
        if len(interpOpts)==0:  #i.e. for OpenFOAM
           nam=caseName+'_'+str(i+1)
           db1=ofProfsReader.dbCnstrctr_OF(prePath,nam)
        else:    #i.e. Nek5000
           nam=caseName+'_'+str(i+1)
           db1=nekProfsReader_tChan.dbCnstrctr_NEK(prePath+nam,interpOpts)
        db1.update({'name':nam})        
        db.append(db1)
    #(c) Sort list of profile databases in accordance with db_info + merge these two db's
    print('... sorting the read-in databases according to our tensor-product convention.')
    db=nekProfsReader_tChan.sort_merge_dbs(db,db_info)
    return db,nSamples
#
