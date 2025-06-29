# SPDX-FileCopyrightText: 2025 ORDeC contributors
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import partial
from public import populate_all

from .rational import R
from .geoprim import *
from .ordb import *

class PinType(Enum):
    In = 'in'
    Out = 'out'
    Inout = 'inout'

    def __repr__(self):
        return f'{self.__class__.__name__}.{self.name}'

# Misc
# ----

class PolyVec2R(Node):
    ref    = LocalRef('SymbolPoly|SchemWire')
    "This is the polygon"
    order   = Attr(int)
    pos     = Attr(Vec2R)

    ref_idx = Index(ref, sortkey=lambda node: node.order)

# Symbol
# ------

class Symbol(SubgraphHead):
    outline = Attr(Rect4R)
    caption = Attr(str)
    cell = Attr('Cell')

    @cursormethod
    def portmap(cursor, **kwargs):
        def inserter_func(main, sgu):
            main_nid = main.set(symbol=cursor.subgraph).insert(sgu)
            for k, v in kwargs.items():
                SchemInstanceConn(ref=main_nid, here=v.nid, there=cursor[k].nid).insert(sgu)
            return main_nid
        return inserter_func

    @cursormethod
    def _repr_html_(cursor):
        from .render import render_svg
        return render_svg(cursor.subgraph).as_html()

class Pin(Node):
    pintype = Attr(PinType, default=PinType.Inout)
    pos     = Attr(Vec2R)
    align   = Attr(D4, default=D4.R0)
 
class SymbolPoly(Node):
    def __new__(cls, vertices:list[Vec2R]=None, **kwargs):
        main = super().__new__(cls, **kwargs)
        if vertices == None:
            return main
        else:
            def inserter_func(sgu):
                main_nid = main.insert(sgu)
                for i, v in enumerate(vertices):
                    PolyVec2R(ref=main_nid, order=i, pos=v).insert(sgu)
                return main_nid
            return FuncInserter(inserter_func)

    @cursormethod
    @property
    def vertices(cursor):
        return cursor.subgraph.all(PolyVec2R.ref_idx.query(cursor.nid))

class SymbolArc(Node):
    pos         = Attr(Vec2R)
    "Center point"
    radius      = Attr(R)
    "Radius of the arc."
    angle_start = Attr(R, default=R(0))
    "Must be less than angle_end and between -1 and 1, with -1 representing -360° and 1 representing 360°."
    angle_end   = Attr(R, default=R(1))
    "Must be greater than angle_start and between -1 and 1, with -1 representing -360° and 1 representing 360°."
    

# # Schematic
# # ---------

class Net(Node):
    pin = ExternalRef(Pin, of_subgraph=lambda c: c.subgraph.symbol)

class Schematic(SubgraphHead):
    symbol = Attr(Symbol) # Subgraph reference
    outline = Attr(Rect4R)
    cell = Attr('Cell')
    default_supply = LocalRef(Net)
    default_ground = LocalRef(Net)

    @cursormethod
    def _repr_html_(cursor):
        from .render import render_svg
        return render_svg(cursor.subgraph).as_html()

class SchemPort(Node):
    ref = LocalRef(Net)
    ref_idx = Index(ref)
    pos = Attr(Vec2R)
    align = Attr(D4, default=D4.R0)

class SchemWire(SymbolPoly):
    ref = LocalRef(Net)
    ref_idx = Index(ref)

class SchemInstance(Node):
    pos = Attr(Vec2R)
    orientation = Attr(D4, default=D4.R0)
    symbol = Attr(Symbol) # Subgraph reference

    def __new__(cls, connect=None, **kwargs):
        main = super().__new__(cls, **kwargs)
        if connect == None:
            return main
        else:
            return FuncInserter(partial(connect, main))

    @cursormethod
    def loc_transform(cursor):
        return TD4(transl=cursor.pos) * cursor.orientation.value

    @cursormethod
    @property
    def conns(cursor):
        return cursor.subgraph.all(SchemInstanceConn.ref_idx.query(cursor.nid))

class SchemInstanceConn(Node):
    ref = LocalRef(SchemInstance)
    ref_idx = Index(ref)

    here = LocalRef(Net)
    there = ExternalRef(Pin, of_subgraph=lambda c: c.ref.symbol) # ExternalRef to Pin in SchemInstance.symbol

    ref_pin_idx = CombinedIndex([ref, there], unique=True)

class SchemTapPoint(Node):
    ref = LocalRef(Net)
    ref_idx = Index(ref)

    pos = Attr(Vec2R)
    align = Attr(D4, default=D4.R0)

class SchemConnPoint(Node):
    ref = LocalRef(Net)
    ref_idx = Index(ref)

    pos = Attr(Vec2R)

# Simulation hierarchy
# --------------------

class SimNet(Node):
    trans_voltage = Attr(list[float])
    trans_current = Attr(list[float])
    dc_voltage = Attr(float)
    dc_current = Attr(float)

    ref = Attr(type=Net|Pin)

class SimInstance(Node):
    is_leaf = False
    ref = Attr(SchemInstance)

class SimHierarchy(SubgraphHead):
    ref = Attr(Schematic)
    cell = Attr('Cell')

# Every class defined in this file is public:
populate_all()
