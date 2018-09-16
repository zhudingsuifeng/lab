#!/usr/bin/env python3
#coding=utf-8

#import matplotlib
#import pandas
import pygal.maps.world
#from mpl_toolkits.basemap import Basemap

worldmap=pygal.maps.world.World()
worldmap.title="CS"
worldmap.add('',{'cn':0})
worldmap.render_to_png('/home/fly/works/worldmap.png')
worldmap.render()
print("success")
