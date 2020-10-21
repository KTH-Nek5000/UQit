##########################################################################
# Construct databases from the QoIs of turbulent channel flow simulations
##########################################################################
# Saleh Rezaeiravesh,  salehr@kth.se
#-------------------------------------------------------------------------
#
import nekProfsReader_tChan
#
def dbMakerCase_multiPar(prePath,caseName,interpOpts):
    """ 
    Make a sorted list of databases (dicts) for a set of channel flow simulations \
            which are conducted in a computer experiment conducted for two or more uncertain parameters.
       * CFD simulations can be done by either Nek5000 or OpenFOAM.   
       * The database of each simulation is a dict containing profiles of channel flow QoIs which are averaged over time and space.   
       * Each channel flow simulation corresponds a unique sample of uncertain parameters.
         We assume tensor-product over parameter space when designing the experiment.
       * Sorting is required to make sure our convention for tensor-product parameter samples is in place. 
         Convention: Higher the parameter index, slower the loop changes: Fortran-like loop order.

    Args:
       `PrePath`: string
          Path to the channel flow data
       `caseName`: string
          Name of the computer experiment (i.e. the set of channel flow simulations)
       `interOpts`: dict
          Options for interpolating the Nek5000 data from GLL points to arbitrary uniform mesh. 
          for OpenFoam, interOpts={}
    
    Returns:
       `db`: List of dicts
          Containing the QoIs of channel simulations in the computer experiment
    """
    db_info=nekProfsReader_tChan.sortedCaseInfo(prePath,caseName)
    nSamples=db_info['nSamples']   
    nSamplesProd=1
    for i in range(len(nSamples)):
        nSamplesProd*=nSamples[i]
    db=[]
    for i in range(nSamplesProd):
        if len(interpOpts)==0:
           nam=caseName+'_'+str(i+1)
           db1=ofProfsReader.dbCnstrctr_OF(prePath,nam)
        else:    
           nam=caseName+'_'+str(i+1)
           db1=nekProfsReader_tChan.dbCnstrctr_NEK(prePath+nam,interpOpts)
        db1.update({'name':nam})        
        db.append(db1)
    db=nekProfsReader_tChan.sort_merge_dbs(db,db_info)
    return db,nSamples
#
