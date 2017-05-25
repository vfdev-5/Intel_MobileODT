
# this your custom configuration module
from PyQt4.Qt import *
from PyQt4.QtGui import QPen
import sloth


def BlueRectItem(*args, **kwargs):
    blueRectItem = sloth.items.RectItem(*args, **kwargs)
    blueRectItem.setColor(Qt.blue)       
    return blueRectItem
 

def GreenPointItem(*args, **kwargs):
    greenPointItem = sloth.items.PointItem(*args, **kwargs)
    greenPointItem.setPen(QPen(Qt.green, 30))
    return greenPointItem


LABELS = (
    {"attributes": 
         {"type": "rect",
          "class": "os"},
     "item": "sloth.items.RectItem",
     "inserter": "sloth.items.RectItemInserter",
     "hotkey": 'q',
     "text": "Os"   
    },
    {"attributes": 
         {"type": "rect",
          "class": "cervix"},
     "item": BlueRectItem,
     "inserter": "sloth.items.RectItemInserter",
     "hotkey": 'w',
     "text": "Cervix"
    },
    {"attributes": 
         {"type": "point",
          "class": "to_remove"},
     "item": "sloth.items.PointItem",
     "inserter": "sloth.items.PointItemInserter",
     "hotkey": 'r',
     "text": "Remove"
    },
    {"attributes":
         {"type": "point",
          "class": "ok"},
     "item": GreenPointItem,
     "inserter": "sloth.items.PointItemInserter",
     "hotkey": 'e',
     "text": "OK"
     },

)
