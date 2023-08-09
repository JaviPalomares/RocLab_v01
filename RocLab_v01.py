# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:10:50 2023

@author: jpalomares
"""

import numpy as npy
from matplotlib import pyplot as plt, patches
import math
import streamlit as st
import mpld3
import streamlit.components.v1 as components

########################################################################
###############----DEFINIMOS FUNCIONES PREVIAS----######################
########################################################################

def HB_parameters(GSI, mi, D):
    #Definimos los parámetros característicos del macizo          
    mb=mi*math.exp((GSI-100)/(28-14*D))
    s=math.exp((GSI-100)/(9-3*D))
    a=0.5+1/6*(math.exp(-GSI/15)-math.exp(-20/3)) 
    
    return mb, s,a


#Función criterio de rotura HB
def HB_failure(s1,s3,UCS,GSI,mi,D):
    
    mb,s,a=HB_parameters(GSI,mi,D)
    sigma=[-UCS/mb*s]
    tau=[0]
    for i in npy.linspace(-UCS/mb*s+0.001,s1,100):
        sigma3=i
        sigma1=sigma3+UCS*(mb*sigma3/UCS+s)**a
        delta=1+mb*a*(mb*sigma3/UCS+s)**(a-1)
        sigma.append((sigma1+sigma3)/2-(sigma1-sigma3)/2*(delta-1)/(delta+1))
        tau.append((sigma1-sigma3)*math.sqrt(delta)/(delta+1))
           
    return sigma,tau

#Función rotura estricta por GSI de HB
def HB_failure_strict_GSI(s1,s3,UCS,GSI,mi,D):
    
    delta=2
    for n in range(200):
        if delta>0:
            GSI=GSI+0.5
        else:
            GSI=GSI-0.5
            
        mb,s,a=HB_parameters(GSI,mi,D)
        sigma1=s3+UCS*(mb*s3/UCS+s)**a
        delta=s1-sigma1
        if GSI==100:
            break
        
    GSI=round(GSI,0)
          
    return GSI

#Fórmula para obtener parámetros equivalentes entre MC y HB
def MC_eq_HB(UCS,GSI,mi,D,sigma3_max):
    
    #Definimos los parámetros característicos del macizo          
    mb=mi*math.exp((GSI-100)/(28-14*D))
    s=math.exp((GSI-100)/(9-3*D))
    a=0.5+1/6*(math.exp(-GSI/15)-math.exp(-20/3))
    traccion= -UCS/mb*s
    
    s3n=sigma3_max/UCS
    
    phi=math.asin((6*a*mb*(s+mb*s3n)**(a-1))/(2*(1+a)*(2+a)+6*a*mb*(s+mb*s3n)**(a-1)))
    coh=UCS*(((1+2*a)*s+(1-a)*mb*s3n)*(s+mb*s3n)**(a-1))/((1+a)*(2+a)*math.sqrt(1+(6*a*mb*(s+mb*s3n)**(a-1)/((1+a)*(2+a)))))
    
    
    return phi, coh, traccion


#Fórmula pata dibujar la envolvente de MC
def MC_failure(phi,coh,traccion,stop):
    
    sigma=[traccion]
    tau=[0]
    
    for i in npy.linspace(traccion,stop,100):
        sigma.append(i)
        tau.append(coh+(i-traccion)*math.tan(phi))
    
    return sigma, tau
 
#Fórmula para obtener la sigma3máx   
def sigma3_max(Tipo,gamma,H,UCS,GSI,mi,D):
    
    mb,s,a=HB_parameters(GSI,mi,D)       
    sigma_cm=UCS*((mb+4*s-a*(mb-8*s))*((mb/4+s)**(a-1)))/(2*(1+a)*(2+a))
    if Tipo==0:
        sigma_max=sigma_cm*0.47*(sigma_cm/gamma/H)**(-0.94)
    
    else:
        sigma_max=sigma_cm*0.72*(sigma_cm/gamma/H)**(-0.91)
    
    return sigma_max

    
########################################################################
################----GENERAMOS APP DE STREAMLIT----######################
########################################################################  
  
st.header('Bienvenid@ a RocLab_v01 - by: jpalomares')
st.caption('Esta aplicación permite obtener e factor de seguridad de un terreno tipo Hoek-Brown variando únicamente su GSI. Para ello es necesario definir las propiedades del macizo rocos y su estado tensional. El resultado se obtiene en forma de un gráfico donde se incluye toda la información relevante.')
st.sidebar.header('Parametros de entrada')
with st.sidebar:
    s1=st.number_input("Introduce tensión principal mayor [kPa]", step=50, value=3500)
    s3=st.number_input("Introduce tensión principal menor [kPa]", step=50, value=900)
    UCS=st.number_input("Introduce valor resistencia compresión simple [MPa]",step=0.5, value=20.0)*1000
    mi=st.number_input("Introduce valor parámetro m", min_value=1, max_value=50, step=1, value=7)
    D=st.number_input("Introduce parametro de alteración", min_value=0.0, max_value=1.0, step=0.1)
    GSI=st.slider("Introduce GSI", min_value=0, max_value=100, step=1, value=50)
    

Eq_option=st.checkbox('Mostrar criterios equivalentes de Mohr Coulomb', value=False)
FoS_option=st.checkbox('Calcular envolvente con GSI crítico', value=True) 


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = False

fig,ax = plt.subplots()
   
center=(s1+s3)/2
rad=abs(s1-s3)/2

circle = patches.Wedge((center, 0), rad, 0, 180) 
circle.set(color='blue', width=0.01, fill=True)   
ax.add_patch(circle)
ax.axis('equal')     

#Representamos el criterio de rotura especificado
[sigma,tau]=HB_failure(s1,s3,UCS,GSI,mi,D)
ax.plot(sigma,tau,label=('HB para GSI='+str(GSI)),color='black')

if Eq_option:
    with st.sidebar:
        ajuste=st.radio('Elegir tipo de ajuste según el nivel de confinamiento',('Túnel','Talud'),horizontal=True)
        if ajuste=='Túnel':
            Tipo=0
            H=st.number_input("Introduce profundidad media del túnel [m]", step=1, value=50)
            gamma=st.number_input("Introduce peso específico del terreno", step=0.5, value=26.0)
        else:
            Tipo=1
            H=st.number_input("Introduce altura del talud [m]", step=1, value=50)
            gamma=st.number_input("Introduce peso específico del terreno", step=0.5, value=26.0)
    
    #Representamos el criterio de rotura equivalente MC
    sigma_max=sigma3_max(Tipo, gamma, H, UCS,GSI,mi,D)
    phi,coh,traccion=MC_eq_HB(UCS,GSI,mi,D, sigma_max)
    [sigma,tau]=MC_failure(phi, coh, traccion, UCS)
    ax.plot(sigma,tau,label=('MC equivalente'),linestyle='dashed',color='black') 

    
if FoS_option:
    GSI_strict=HB_failure_strict_GSI(s1,s3,UCS,GSI,mi,D)
    st.caption('El factor de seguridad del terreno es '+str(round(GSI/GSI_strict,2)))
    [sigma,tau]=HB_failure(s1,s3,UCS,GSI_strict,mi,D)
    ax.plot(sigma,tau, label=('Rotura para GSI='+str(int(GSI_strict))),color='red')
    if Eq_option:
        #Representamos el criterio de rotura equivalente MC
        sigma_max=sigma3_max(Tipo, gamma, H, UCS,GSI_strict,mi,D)
        phi,coh,traccion=MC_eq_HB(UCS,GSI_strict,mi,D,sigma_max)
        [sigma,tau]=MC_failure(phi, coh,traccion, UCS)
        ax.plot(sigma,tau,label=('MC equivalente'),linestyle='dashed',color='red')

ax.set_title('ESTADO TENSIONAL DEL TERRENO')
ax.set_ylabel('Tensión cortante [kPa]')
ax.set_xlabel('Tensión normal [kPa]')    
ax.legend(loc='upper right')
plt.xlim([0, s1*1.2])
plt.ylim([-0, (s1-s3)/2*1.7])

fig_html=mpld3.fig_to_html(fig)
components.html(fig_html, height=400)


col1,col2=st.columns(2)

with col1:
    st.caption('Parametros para GSI='+str(GSI))
    mb,s,a=HB_parameters(GSI,mi,D)
    st.caption('Hoek-Brown :  mb='+str(round(mb,3))+' / s='+str(round(s,3))+' / a='+str(round(a,3)))
    if Eq_option:
        phi,coh,traccion=MC_eq_HB(UCS,GSI,mi,D, sigma_max)
        st.caption('Mohr-Coulomb :  phi='+str(round(math.degrees(phi),2))+' / coh='+str(round(coh,2)))

if FoS_option:
    with col2:
        st.caption('Parametros para GSI='+str(GSI_strict))
        mb,s,a=HB_parameters(GSI_strict,mi,D)
        st.caption('Hoek-Brown :  mb='+str(round(mb,3))+' / s='+str(round(s,3))+' / a='+str(round(a,3)))
        if Eq_option:
            phi,coh,traccion=MC_eq_HB(UCS,GSI_strict,mi,D, sigma_max)
            st.caption('Mohr-Coulomb :  phi='+str(round(math.degrees(phi),2))+' / coh='+str(round(coh,2)))
    
