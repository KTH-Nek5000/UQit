##################################################
# Assign LaTex values to strings
##################################################
#
def texLabel_constructor():
    """
    LaTex labels for quantities appearing in UQ plots.
    """
    texLab={'uTau':r'$\langle u_\tau \rangle$'}
    texLab.update({'y':r'$y$'})
    texLab.update({'y+':r'$y^*$'})
    texLab.update({'u':r'$\langle u \rangle$'})
    texLab.update({'u+':r'$\langle u \rangle^*$'})
    texLab.update({"u'":r'$u^\prime_{\rm rms}$'})
    texLab.update({"u'+":r'$u^{\prime^*}_{\rm rms}$'})
    texLab.update({"v'":r'$v^\prime_{\rm rms}$'})
    texLab.update({"v'+":r'$v^{\prime^*}_{\rm rms}$'})
    texLab.update({"w'":r'$w^\prime_{\rm rms}$'})
    texLab.update({"w'+":r'$w^{\prime^*}_{\rm rms}$'})
    texLab.update({"uv":r'$\langle u^\prime v^\prime \rangle$'})
    texLab.update({"uv+":r'$\langle u^\prime v^\prime \rangle^*$'})
    texLab.update({"tke":r'$\mathcal{K}$'})
    texLab.update({"tke+":r'$\mathcal{K}^*$'})
    texLab.update({'dy+1':r'$\Delta y^+_1$'})
    texLab.update({'dx+':r'$\Delta x^+$'})
    texLab.update({'dz+':r'$\Delta z^+$'})
    texLab.update({'dUc':r'$\epsilon[\langle u_c \rangle ]$'})
    texLab.update({'duTau':r'$\epsilon[\langle u_\tau \rangle ]$'})
    texLab.update({'dU_lInf':r'$\epsilon_\infty[\langle u\rangle]$'})
    texLab.update({'duv_lInf':r'$\epsilon_\infty[\langle u^\prime v^\prime \rangle]$'})
    texLab.update({'durms_lInf':r'$\epsilon_\infty[u^\prime_{\rm rms}]$'})
    texLab.update({'dvrms_lInf':r'$\epsilon_\infty[v^\prime_{\rm rms}]$'})
    texLab.update({'dwrms_lInf':r'$\epsilon_\infty[w^\prime_{\rm rms}]$'})
    texLab.update({'dK_lInf':r'$\epsilon_\infty[\mathcal{K}]$'})
    texLab.update({'cutoffRatio':r'$\lambda$'})
    texLab.update({'explicitWeight':r'$\alpha$'})
    texLab.update({'hpfrtWeight':r'$\chi$'})
    return texLab 
#
def texLabel(qoiName):
    """
    look up the LaTex label corresponding to qoiName
    """
    texLab_db=texLabel_constructor()
    return(texLab_db[qoiName])
#
