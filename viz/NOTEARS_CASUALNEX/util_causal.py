# -*- coding: utf-8 -*-



from causalnex.structure import StructureModel
sm = StructureModel()




sm.add_edges_from([
    ('health', 'absences'),
    ('health', 'G1')
])









