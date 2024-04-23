# Import necessary libraries
from sentence_transformers import SentenceTransformer
import openai
import numpy as np

# Initialize the models
openai.api_key = 'k-proj-YZ8xtTgqkp3tsj5sf6UST3BlbkFJh6wMOocUFx7DGIne24Gt'
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the document text
document_text = """
The unofficial consolidated version of the Agricultural Land Act comprises:
-         Agricultural Land Act – ZKZ (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 59/96 of 25 October 1996),
-         Decision abrogating the provision of paragraph two of Article 124 of the Agricultural Land Act (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 31/98 of 18 April 1998),
-         Replacement of the Retail Price Index by the Cost of Living Index Act – ZNIDC (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 1/99 of 9 January 1999),
-         Agriculture Act – ZKme (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 54/00 of 16 June 2000),
-         Decision abrogating the first sentence of paragraph two of Article 115 of the Agricultural Land Act and abrogating part of the second sentence of paragraph two of Article 115 of the same Act (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 68/00 of 31 July 2000),
-         Decision abrogating Chapter III of the Agricultural Land Act with a suspensive deadline of one year (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 68/00 of 31 July 2000),
-         Marine Fisheries Act – ZMR-1 (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 58/02 of 4 July 2002),
-         Act Amending the Agricultural Land Act – ZKZ-A (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 67/02 of 26 July 2002),
-         Spatial Management Act – ZUreP-1 (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 110/02 of 18 December 2002),
-         Construction Act – ZGO-1 (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 110/02 of 18 December 2002),
-         Act Amending the Agricultural Land Act – ZKZ-B (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 36/03 of 16 April 2003),
-         Agricultural Land Act – Official Consolidated Text – ZKZ-UPB1 (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 55/03 of 9 June 2003),
-         Act Amending the Agricultural Land Act – ZKZ-C (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 43/11 of 2 June 2011),
-         Agricultural Land Act – Official Consolidated Text – ZKZ-UPB2 (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 71/11 of 9 September 2011),
-         Act Amending the Agricultural Land Act – ZKZ-D (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 58/12 of 31 July 2012),
-         Act Amending the Agricultural Land Act – ZKZ-E (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 27/16 of 11 April 2016),
-         Act Amending the Agriculture Act – ZKme-1D (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 27/17 of 2 June 2017),
-         Act Amending the Agricultural Land Act – ZKZ-F (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 79/17 of 28 December 2017).
AGRICULTURAL LAND ACT
(ZKZ)
(Unofficial consolidated version No. 17)
I. GENERAL PROVISIONS
Article 1
This Act regulates the protection and management of agricultural land by laying down its classification, use and cultivation, agricultural land transactions and lease arrangements, agricultural operations and common pasture.
The provisions of this Act shall also apply mutatis mutandis to forests unless otherwise provided by an Act.
Article 1a
The goals of this Act shall be the following:
-         to preserve and improve production potential and increase agricultural land area intended for food production;
-         to foster the sustainable management of fertile soil;
-         to foster landscape preservation and preserve and develop rural areas.
Article 1b
In order to achieve the goals referred to in the preceding Article, the Government of the Republic of Slovenia (hereinafter: Government) shall adopt agricultural land policy measures.
The subject of agricultural land policy measures shall be agricultural land as referred to in paragraph two of Article 2 of his Act.
The funds for drawing up and implementing agricultural land policy measures shall be provided from:
-         compensation due to the conversion of agricultural land to other uses;
-         other budgetary funds of the Republic of Slovenia;
-         other sources.
Agricultural land policy measures shall address the overgrowth of agricultural land and the implementation of agricultural operations.
The Agency of the Republic of Slovenia for Agricultural Markets and Rural Development shall be responsible for the implementation of the measures referred to in the preceding paragraph. For these measures, the provisions of the Act regulating agriculture shall apply to beneficiaries, while the provisions of the aforementioned Act that regulate the procedure for the implementation of rural development policy measures shall apply to the procedure for the implementation of measures.
The measures referred to in paragraph four of this Article may also be carried out by the Farmland and Forest Fund of the Republic of Slovenia (hereinafter: Fund); in addition, the purchase of agricultural land by the Fund shall also be deemed to be an agricultural land policy measure. The Fund shall carry out these measures in accordance with the approved annual work programme. The funds referred to in paragraph four of this Article acquired by the Fund for the implementation of the measures referred to in this paragraph shall be included in the funds acquired by the Fund in accordance with the Act regulating the Farmland and Forest Fund of the Republic of Slovenia.
Article 1ba
The funds collected from compensation due to the conversion of agricultural land to other uses may be appropriated for the funding of the construction, remediation or modernisation of water reservoirs that have the status of water infrastructure in accordance with the regulations governing waters and are also intended for the irrigation of agricultural land.
The amount of funding as well as mutual rights, obligations and deadlines shall be stipulated by a contract between the ministry responsible for agriculture and the person investing in works on the water reservoir referred to in the preceding paragraph. The percentage of funds allocated for the works referred to in the preceding paragraph may not exceed the percentage of the volume of the water reservoir that is intended for the irrigation of agricultural land.
Article 1c
For the drawing up of the background documents referred to in paragraph two of Article 3c of this Act, the study referred to in paragraph eight of Article 3c of this Act and the analyses referred to in paragraphs five and six of Article 3e of this Act, and for the drawing up of proposals for the introduction of agricultural operations and studies related thereto, the organisations referred to in Article 3f of this Act, the producers and drafters of spatial planning documents and the entities proposing agricultural operations shall obtain data from official databases, except personal data.
For the drawing up of proposals for the introduction of agricultural operations and other studies related thereto, entities proposing agricultural operations may, in addition to the data referred to in the preceding paragraph, obtain and further process the following personal data from the land register: personal name, residential address and unique personal identification number (hereinafter: UPIN) of owners or lessees.
For the purpose of drawing up the study referred to in paragraph eight of Article 3c of this Act and the analyses referred to in paragraph six of Article 3e of this Act, producers and drafters of spatial documents may obtain and use data on the identification number of agricultural holdings in relation to the corresponding graphical land use units of agricultural holdings from the register of agricultural holdings kept in accordance with the Act regulating agriculture.
The ministry responsible for agriculture, the organisations referred to in Article 3f of this Act, the producers and drafters of spatial documents and entities proposing agricultural operations may obtain data on agricultural land from official databases free of charge.
II. THE CLASSIFICATION, PROTECTION, USE AND CULTIVATION OF AGRICULTURAL LAND
Article 2
Agricultural land under this Act shall mean land suitable for agricultural production that is designated as agricultural land in accordance with paragraphs two and three of this Article.
Agricultural land shall be determined by the spatial planning documents of local communities as areas of agricultural land and shall be classified as either areas of permanently protected agricultural land or areas of other agricultural land.
For the implementation of Articles 4 and 7 of this Act and the implementation of the provisions referring to common pastures, in addition to the land referred to in the preceding paragraph, land designated for non-agricultural use in spatial planning documents of local communities that is classified as arable fields, home gardens, grassland, permanent cropland and other agricultural land according to the records of the actual use of agricultural and forest land (hereinafter: records of actual land use) in accordance with the Act regulating agriculture shall be deemed to be agricultural land.
Article 3
The ministry responsible for agriculture shall be responsible for agricultural land management in spatial planning procedures pursuant to the regulations governing spatial planning.
The ministry responsible for agriculture shall be responsible for land management also in spatial planning procedures where the surface area and the detailed designated use of land that includes agricultural production facilities within the production areas are subject to change.
Article 3a
At the request of the ministry responsible for agriculture, the Fund shall participate in carrying out the duties of the ministry responsible for agriculture in spatial planning procedures.
Article 3b
Taking into account the national strategic spatial planning document, the Government shall designate, by a regulation, the agricultural and food production areas of strategic importance for the Republic of Slovenia due to the production potential of the agricultural land, its surface area, territorial compactness, food production capacity, the preservation of its rural character, rural development and landscape preservation.
The areas referred to in the preceding paragraph shall represent potential areas of permanently protected agricultural land.
Article 3c
Proposals of permanently protected agricultural land areas shall be drawn up on the basis of the surface area and territorial compactness thereof, while taking into account the following conditions:
-         the land rating of the agricultural land pursuant to the regulations governing real estate records (hereinafter: land rating of agricultural land) is higher than 35,
-         the inclination is not more than 11 percent,
-         land consolidation, drainage or irrigation have been carried out,
-         the availability of water sources suitable for irrigation,
-         the existence of permanent cropland, or
-         the local characteristics of agricultural production and use of agricultural land.
For each local community and at the expense of the ministry responsible for agriculture, an organisation referred to in Article 3f of this Act shall draw up a background document in the field of agriculture (hereinafter: background document) that must include findings on compliance with the conditions referred to in the preceding paragraph. At the request of the ministry responsible for agriculture, the background document must include findings on compliance with the conditions referred to in the preceding paragraph also for other land potentially suitable for agricultural production.
In the background document, the organisation referred to in Article 3f of this Act shall, based on the findings referred to in the preceding paragraph, draw up a proposal of areas of:
-         permanently protected agricultural land with regard to the regulation referred to in Article 3b of this Act, and
-         other agricultural land.
The ministry responsible for agriculture shall have background documents drawn up when it is evident from a spatial information system, in accordance with the Act regulating spatial planning, that a decision on the beginning of the procedure for drawing up a spatial planning document has been adopted from which it follows that the spatial planning document proposes a change to the designated use of agricultural land, pursuant to the Act regulating spatial planning.
An organisation referred to in Article 3f of this Act shall draw up the background document referred to in this Article within three months of receiving an order.
The ministry responsible for agriculture shall publish on its website the proposals of areas referred to in paragraph three of this Article.
In drafting a spatial planning document, the local community must take into account the data in the background document and planning shall first take place regarding land designated for non-agricultural use while taking into account the principles of the Act regulating spatial planning. If this is not possible, however, while taking into account the proposal of areas of permanently protected and other agricultural land, planning shall first take place regarding areas of the proposed other agricultural land and only subsequently regarding areas of the proposed permanently protected agricultural land; for such areas, planning shall first take place regarding agricultural land with a lower land rating.
The local community must attach to the draft spatial planning document a study of developments on agricultural land showing that the planned developments are in accordance with the preceding paragraph.
The spatial planning document of a local community shall determine areas of permanently protected and other agricultural land.
The minister responsible for agriculture shall determine the detailed content of the background documents referred to in paragraph two of this Article.
The minister responsible for agriculture, in agreement with the minister responsible for spatial planning, shall determine the detailed content of the study referred to in paragraph eight of this Article.
The minister responsible for agriculture shall determine the detailed conditions referred to in paragraph one of this Article.
Article 3č
In its spatial planning document, a local community may allow within agricultural land areas the construction of facilities or developments as follows:
a)   	agricultural operations and water reservoirs for the purpose of irrigating agricultural land;
b)   	simple and non-complex auxiliary agricultural and forestry facilities in accordance with the regulation governing the classification of structures with regard to their complexity, except cellars and wine cellars;
c)   	structures that are products placed on the market pursuant to the regulation governing technical requirements for products and conformity assessment and which may be, in accordance with the regulation governing the classification of structures with regard to their complexity, classified as auxiliary agricultural and forestry facilities, except cellars and wine cellars, whose size shall not exceed the size of non-complex structures, except built greenhouses, which may exceed the size of non-complex structures;
č)	beehives, i.e. wooden, single-storey, ground-level structures on point foundations, intended for beekeeping, with a floor area up to and including 40 m2;
d)   	sheds, i.e. wooden, single-storey, ground level structures on point foundations, intended for sheltering grazing farm animals, with a floor area up to and including 100 m2;
e)   	auxiliary agricultural and forestry equipment (e.g. grape vine trellises, wind rattles, posts, poles, wire supports, hail protection net supports, bird protection net supports, pens, livestock gazing fences, fences and supports for permanent crops, fences for the protection of crops, movable tunnels and canopies, protective netting);
f)         auxiliary structures for monitoring the state of the environment and natural phenomena;
g)   	research on subterranean waters, mineral resources and geothermal energy sources;
h)   	temporary structures and temporary activities during events or seasons:
-  a stage with a roof made of prefabricated components,
-  a circus if the tent and other structures are prefabricated,
-  a temporary open air gallery for an audience,
-  movable wooden structures for animal rearing (e.g. movable beehives, movable hen houses, movable rabbit houses);
i)         hides, i.e. wooden structures without foundations (hunting blinds, bird observatories);
j)         provisional arrangements for the purpose of defence and protection against natural and other disasters;
k)   	access to a facility complying with a spatial planning document in the case of a facility that:
-  is allowed to be built on agricultural land,
-  is recognised as a dispersed construction (land underneath buildings outside areas of construction land), or
-  is allowed to be built within the area of a dispersed settlement;
l)         civil engineering facilities that are classified into two groups according to the regulations on the introduction and application of the uniform classification of facilities and on the designation of facilities of national importance:
-  long distance pipelines, long distance (backbone) communication networks and long distance power transmission lines, along with the associated facilities and connections thereto, and
-  local pipelines, local (distribution) power lines and local (access) communication networks, along with the associated facilities and connections thereto;
m) 	the reconstruction of municipal and state roads in accordance with the Act regulating roads. Also permitted are facilities required for the planned reconstruction of roads (e.g. roofed bus stop shelters, cycling trails and hiking trails, retaining and support walls, overpasses, underpasses, culverts, noise barriers, auxiliary road structures, urban equipment) and public utility infrastructure structures that must be built or relocated along a road due to road reconstruction;
n)   	small wind farms with a nominal power up to 1 MW, in the case of agricultural land with a land rating of less than 35.
Notwithstanding point b) of the preceding paragraph, in its spatial planning document a local community may allow only the following simple and non-complex auxiliary agricultural and forestry facilities to be built in areas with permanently protected agricultural land: greenhouses, fences for livestock grazing, game pens, fences and supports for permanent crops, supports for anti-hail nets, fences for crop protection, hayracks, double hayracks, milking parlours, watering troughs, and farm sheds.
Notwithstanding point c) of paragraph one of this Article, in its spatial planning document a local community may only allow the following facilities that are products placed on the market pursuant to the regulation governing technical requirements for products and conformity assessment and whose size does not exceed the size of non-complex structures, except built greenhouses, which may exceed the size of non-complex structures, to be built in areas of permanently protected agricultural land: greenhouses, fences for livestock grazing, game pens, fences and supports for permanent crops, supports for anti-hail nets, fences for crop protection, hayracks, double hayracks, milking parlours, watering troughs, and farm sheds.
In its spatial planning document, a local community may prescribe conditions stricter than those laid down in Article 3ča of this Act regarding the required surface area of agricultural land that must be met by investors in order to build the sheds referred to in point d) of paragraph one of this Article, auxiliary agricultural and forestry facilities referred to in points b) and c) of paragraph one of this Article, and structures referred to in paragraphs two and three of this Article that are classified as non-complex structures pursuant to the regulation governing the classification of structures with regard to their complexity, except greenhouses, fences for livestock grazing, game pens, fences and supports for permanent crops and supports for hail nets and fences for the protection of agricultural products.
In addition to the conditions referred to in the preceding paragraph, in its spatial planning document a local community may prescribe additional conditions and criteria that must be met by investors in order to build on agricultural land, which shall apply to the construction of structures referred to in paragraph one of this Article, except provisional arrangements for the purpose of defence and protection against natural and other disasters.
On permanently protected agricultural land it shall not be permitted to establish areas for mitigation and compensatory measures under the Act regulating nature conservation.
Notwithstanding the preceding Article, areas for mitigation or compensatory measures under the Act regulating nature conservation that are related to existing or planned spatial developments of national importance in the field of road and railway infrastructure referred to in paragraph one or two of Article 3e of this Act may be exceptionally established on permanently protected agricultural land if they cannot be located on other land.
The auxiliary agriculture and forest facilities referred to in points b) and c) of paragraph one of this Article and paragraphs two and three of this Article, the beehives referred to in point č) of paragraph one of this Article and the sheds referred to in point d) of paragraph one of this Article may be used exclusively for agricultural purposes.
The minister responsible for agriculture shall, in accordance with the minister responsible for defence, specify in further detail the types of provisional arrangements for the purpose of defence and protection against natural and other disasters referred to in point j) of paragraph one of this Article.
Article 3ča
Notwithstanding the Act regulating construction, sheds and auxiliary agriculture and forest facilities that are classified as non-complex structures pursuant to the regulation governing the classification of structures with regard to their complexity, except greenhouses, fences for livestock grazing, game pens, fences and supports for permanent crops and supports for hail nets and fences for the protection of agricultural products, shall be built on agricultural land by investors who are owners or lessees of:
a)   	at least 1 ha of land classified as arable fields, home gardens, grassland, permanent cropland and other agricultural land according to the records of actual land use, or
b)   	at least 5,000 m2 of land classified as permanent cropland according to the records of actual land use.
Article 3d
Areas of permanently protected agricultural land may not be changed for at least ten years from the entry into force of the spatial planning document of a local community by which such areas were determined.
Notwithstanding the preceding paragraph, the following facilities may be planned within areas of permanently protected agricultural land in the spatial planning documents of local communities by converting the agricultural use of land to other uses before the expiration of the period referred to in the preceding paragraph:
-         road infrastructure of local importance necessary for connection to the road infrastructure referred to in paragraph one or two of Article 3e of this Act,
-         the partial or complete relocation of agricultural holdings related to the planning of the road infrastructure referred to in paragraph one or two of Article 3e of this Act,
-         water infrastructure and water regulation developments of vital importance for protection against the harmful effects of water as well as the water infrastructure and water regulation developments of local importance referred to in paragraph one or two of Article 3e of this Act.
A local community must first plan the developments referred to in the preceding paragraph on land designated for non-agricultural use. If this is not possible, they must first be planned in areas of other agricultural land and only afterwards in areas of permanently protected agricultural land; in these areas, the planning shall first take place regarding agricultural land with a lower land rating.
In the spatial planning procedure of local communities, areas of other agricultural land may also be converted before the expiration of the period referred to in paragraph one of this Article. In such cases, the principle of the Act regulating spatial planning must be taken into account and the planning shall first take place regarding land designated for non-agricultural use. Where this is not possible, the planning shall first take place regarding areas of other agricultural land with a lower land rating.
The local community must attach to the draft spatial planning document referred to in this Article a study of developments on agricultural land showing that the developments are being planned in accordance with paragraph three of this Article or the preceding paragraph.
After the termination of the period referred to in paragraph one of this Article, the conversion of areas of permanently protected agricultural land and areas of other agricultural land may be carried out according to the procedure referred to in Article 3c of this Act, whereby an organisation referred to in Article 3f of this Act shall verify the data contained in the background document.
The minister responsible for agriculture, in agreement with the minister responsible for spatial planning, shall specify the detailed content of the study referred to in paragraph five of this Article.
Article 3e
Notwithstanding the provisions of paragraph one of the preceding Article, the expansion of areas of existing development projects of national importance may be planned within areas of permanently protected agricultural land before the termination of the period referred to in paragraph one of the preceding Article.
Notwithstanding the provisions of paragraph one of the preceding Article, the following new development projects of national importance in the following fields may be planned within areas of permanently protected agricultural land prior to the termination of the period referred to in paragraph one of the preceding Article:
a)   	road infrastructure;
b)   	railway infrastructure;
c)   	energy infrastructure for electricity supply, as follows:
-  hydroelectric power plants, geothermal wells, power lines, natural gas pipelines, hot water pipelines and oil pipelines and connections thereto;
-  wind farms on agricultural land with a land rating up to and including 40;
č)	water infrastructure and water regulation projects;
d)   	defences and protection against natural and other disasters;
e)   	air transport, i.e. radio navigation devices in air traffic control.
Development projects of national importance referred to in paragraphs one and two of this Article shall first be sited on land designated for non-agricultural use. Where this is not possible, development projects of national importance shall, while taking into account areas of permanently protected and other agricultural land, first be sited within areas of other agricultural land and only afterwards within areas of permanently protected agricultural land and within these areas, planning shall first take place regarding agricultural land with a lower land rating. At the same time, existing networks of transport and public utility infrastructure must be taken into account to the greatest extent possible and the planning must be carried out rationally and in a manner enabling the maximum preservation of agricultural land and its territorial compactness.
In preparing an initiative for the drawing up of the national spatial plan, an initiator pursuant to the Act regulating the siting of development projects of national Importance (hereinafter: initiator) must take into account areas of permanently protected and other agricultural land for local communities whose territory is to be included in the plan.
When a development project of national importance is planned in areas of permanently protected and other agricultural land, the producer of the spatial planning document for the preparation of a study of variants provided by the regulations governing the siting of development projects of national importance shall conduct an analysis of the impact of developments on agricultural land, containing:
-         the extent of the impact of individual variants with respect to the land rating of agricultural land;
-         the extent of the impact of individual variants on the area of agricultural operations that have been carried out;
-         types of developments and their impact on territorially compact agricultural land areas.
In the stage of drafting the national spatial plan, an analysis of the impact of developments on agricultural land shall be conducted for the selected variant, which shall contain the extent of the impact of development projects on graphical units of land use of agricultural holdings for individual agricultural holdings in accordance with the Act regulating agriculture.
Article 3ea
In its municipal detailed spatial plan (hereinafter: MDSP), a local community may, in accordance with the Act regulating spatial planning and provided that this is not in contravention of the strategic spatial development guidelines of the local community, plan the following agricultural facilities directly intended for agricultural activity:
a)   	structures for crop production, if the manner of production is directly linked to agricultural land;
b)       structures for animal husbandry (poultry farms, stalls, piggeries, sheds, stud farms and similar facilities for animal husbandry), including facilities for manure and slurry storage, except facilities for which an environmental impact assessment must be conducted in accordance with the regulation governing activities subject to an environmental impact assessment;
c)       facilities for crop storage (farm silos, granaries, cellars, haylofts, barns, drying frames, corn granaries and similar structures for crop storage, except wine cellars and vineyard houses) and facilities for the processing of own agricultural products (cheese dairies, fruit-drying kilns, etc.);
č)	other non-residential agricultural facilities (structures for the storage of agricultural equipment, tools and machinery).
The facilities referred to in this Article shall be planned in the vicinity of the existing location of the agricultural holding or cooperative registered for agricultural activity; where this is not possible; however, they shall first be planned regarding agricultural land with a lower land rating.
For planning the facilities under this Article, a local community must attach a study to the draft MDSP showing that the facilities referred to in this Article are planned in accordance with the preceding paragraph.
Only initiatives regarding subjects that meet the following conditions may be included in the procedure for the drawing up of a MDSP referred to in this Article:
a)   	in the case of a farm under the Act regulating agriculture, the holder of such a farm must be covered by obligatory pension and disability insurance pursuant to Article 17 of the Pension and Disability Insurance Act (Official Gazette of the Republic of Slovenia [Uradni list RS], Nos 96/12, 39/13, 99/13 – ZSVarPre-C, 101/13 – ZIPRS1415, 44/14 – ORZPIZ206, 85/14 – ZUJF-B, 95/14 – ZUJF-C, 90/15 – ZIUPTD and 102/15);
b)   	in the case of an agricultural holding under the Act regulating agriculture organised as a sole proprietor or a legal entity, such agricultural holding must generate at least 60 percent of its annual income through agricultural activity in the year prior to the submission of the initiative referred to in paragraph five of this Article, whereby such income must exceed EUR 30,000;
c)   	in the case of a cooperative registered for agricultural activity, the cooperative must generate at least EUR 50,000 of its annual income through agricultural activity in the year prior to the submission of the initiative referred to in paragraph five of this Article;
č)	in the case of an agricultural community registered pursuant to the Agricultural Communities Act (Official Gazette of the Republic of Slovenia [Uradni list RS], No. 74/15), such agricultural community must own at least 20 hectares of agricultural land.
An initiative referred to in the preceding paragraph shall be submitted to the local community:
Notwithstanding the provisions of the Act regulating spatial planning, a local community shall, by a decision on the drawing up of the MDSP, verify compliance with the conditions referred to in points a), b), c) and č) of paragraph four of this Article.
When the facilities under this Article are planned regarding agricultural land owned by the Republic of Slovenia, a manager of agricultural land owned by the Republic of Slovenia must agree to the planned facilities.
For the purpose of assessing the public utilities charge, land within the MDSP referred to in this Article shall be deemed to be construction land.
Notwithstanding paragraph one of this Article, the MDSP may be used to plan the relocation of an entire agricultural holding, whereby a permit for the construction of a residential building may, notwithstanding the Act regulating construction, be issued only after obtaining an operating permit for agricultural facilities referred to in paragraph one of this Article.
The agricultural facilities referred to in paragraph one of this Article may be used exclusively for agricultural purposes.
Notwithstanding the Act regulating spatial planning, a local community's spatial planning document that specifies the designated use of land shall be replaced by the MDSP in the part determined by the MDSP upon the MDSP entering into force. On the first occasion of amending the local community's spatial planning document, the local community shall enter the changes that occurred upon the MDSP entering into force into the local community's spatial planning document according to the procedure laid down by the Act regulating spatial planning.
Article 3f
The background documents referred to in Article 3c of this Act shall be drawn up by organisations that meet the prescribed technical, professional and organisational conditions and which are selected by the ministry responsible for agriculture in accordance with the regulations governing public procurement. The ministry responsible for agriculture shall publish the list of organisations on its website.
The minister responsible for agriculture shall specify the technical, professional and organisational conditions referred to in the preceding paragraph.
Article 3g
An investor applying for a building permit for the construction of a facility whose floor area or part thereof lies on agricultural land (hereinafter: floor area for agricultural use) and the land parcel on which the floor area for agricultural use is located has a land rating of more than 50 in accordance with the regulations governing real estate records must pay compensation due to the conversion of agricultural land to other uses (hereinafter: land use conversion compensation).
The floor area referred to in the preceding paragraph shall be deemed to be the gross floor area of a facility at the ground level pursuant to the regulation governing design documentation.
Agricultural land referred to in paragraph one of this Article shall be deemed to be land that, as is evident from the design documentation, is used as arable fields, home gardens, grassland, permanent cropland or other agricultural land.
In addition to the components laid down by the regulation governing design documentation, the design documentation for facilities for which the payment of land use conversion compensation is obligatory must also include data on the type and surface area of the actual use of land on which the floor area of the facility is located, whereby the actual use shall be extracted from the records of actual land use in accordance with the Act regulating agriculture.
Land use conversion compensation shall be calculated using the following equation:
land use conversion compensation =
agricultural land area x A,
whereby the A factor shall be determined on the basis of the land rating of the land on which the facility's layout is located and is registered in the land cadastre, amounting to:
a)   	for the construction of facilities that are, pursuant to regulations on the introduction and application of the uniform classification of facilities, classified in the group of transport infrastructure facilities and other civil engineering facilities,
-  a land rating of 51–60: EUR 1,
-  a land rating of 61–75: EUR 3,
-  a land rating of 76–100: EUR 5;
b)   	for the construction of other facilities:
-  a land rating of 51-60: EUR 4,
-  a land rating of 61-75: EUR 12,
-  a land rating of 76-100: EUR 20.
If the layout of a facility is located on parcels with various land ratings, the corresponding land rating shall be taken into account for each part of the layout.
If the layout of a facility is located on a parcel with a land rating of zero in addition to a land rating of over 50, the land rating of over 50 shall be taken into account for the calculation of land use conversion compensation.
If the layout of a facility for which an application for the issuance of a building permit has been submitted is located on a parcel of land for which the land use conversion compensation that was partially or fully paid had been assessed and levied under previous regulations, land use conversion compensation shall not be assessed for the surface area of the facility's layout located on such parcel.
Notwithstanding the provision of paragraph one of this Article, land use conversion compensation shall not be assessed in procedures for issuing building permits that refer to:
-         the reconstruction or removal of facilities pursuant to the regulations governing construction;
-         the construction of non-complex structures pursuant to the regulations governing the classification of structures with regard to their complexity;
-         changes in the designated use of facilities pursuant to the regulations governing construction;
-         additions to buildings pursuant to the regulations governing construction;
-         the construction of agricultural facilities classified as non-residential agricultural facilities pursuant to regulations on the introduction and application of the uniform classification of facilities and on the designation of facilities of national importance;
-         the construction of civil engineering facilities classified under the group of pipelines, communication networks and electricity lines pursuant to regulations on the introduction and application of the uniform classification of facilities and on the designation of facilities of national importance.
Article 3h
Land use conversion compensation shall be assessed in an administrative procedure ex officio, upon receipt of the application referred to in paragraph one of the preceding Article, by an authority responsible for issuing building permits, which at the same time controls the collection, payment and recording of compulsory charges and other general government revenues in accordance with the regulation governing sub-accounts and the method of payment of compulsory charges and other general government revenues. The ministry responsible for construction shall be competent for the handling of complaints in matters of the assessment of land use conversion compensation at the second instance.
Land use conversion compensation shall be assigned revenue of the budget of the Republic of Slovenia and shall be transferred to the sub-account of general government revenue in accordance with the regulation governing sub-accounts, the method of payment of compulsory charges, and other general government revenues. The funds collected from land use conversion compensation payments shall be allocated for the funding of the preparation and implementation of the agricultural land policy measures referred to in paragraphs four and six of Article 1b of this Act, the operations referred to in Article 1.ba of this Act, the drawing up of the background documents referred to in Article 3c of this Act and other background documents in the field of agricultural land policy and the protection of agricultural land.
The payment of land use conversion compensation is one of the conditions for issuing a building permit in accordance with the Act regulating construction.
For the purpose of monitoring claims, liabilities and payments deriving from compensation, an authority referred to in paragraph one of this Article shall establish and keep records of the following data: the number of decisions on the assessment of land use conversion compensation, the personal name and address or registered office of investors, the parcel number and cadastral municipality of the land parcel to which the land use conversion compensation refers and the amount and date of payment or repayment.
If an investor who paid land use conversion compensation fails to obtain a final building permit or fails to start building the facility for which land use the conversion compensation has been paid, he or she shall have the right to claim a refund thereof within one year of the decision on the refusal to issue a building permit becoming final or the decision to terminate the procedure or upon the expiry of the building permit.
Decisions on the refund of land use conversion compensation under this Article and wrongly paid or overpaid compensation shall be adopted in an administrative procedure by an authority referred to in paragraph one of this Article at the request of the investor or ex officio. The refund of land use conversion compensation shall be covered from assigned funds of the national budget of the Republic of Slovenia.                                                                 	
Article 3ha
Notwithstanding paragraph three of the preceding Article, the condition for issuing a building permit in accordance with the Act regulating construction shall be the payment of half of the assessed land use conversion compensation for facilities of national importance whose investor is the Republic of Slovenia and for which the assessed land use conversion compensation exceeds EUR 2,000,000. The payment of the other half shall be one of the conditions for obtaining an operating permit.
If a facility referred to in the preceding paragraph is a water facility in accordance with the Act regulating waters and is also intended for water abstraction for the irrigation of agricultural land, the land use conversion compensation shall, based on the Government's decision, be reduced by the value of investments in projects intended for water abstraction for the irrigation of agricultural land. This amount shall be deducted from the part of the assessed land use conversion compensation the investor must pay in order to obtain an operating permit, but it shall not exceed half the entire compensation.
Article 3i
Notwithstanding the provision referred to in indent three of paragraph nine of Article 3g of this Act, an investor who applies for a building permit for the construction of an agricultural facility shall be assessed the land use conversion compensation for operations related to changes in the designated use of an agricultural facility referred to in indent five of paragraph nine of Article 3g of this Act for which the land use conversion compensation under this Act has not been paid.
Article 4
Agricultural land must be used in accordance with its designated use and its pollution or other degradation as well as the pollution of plants or any other inhibition of plant growth must be prevented.
Agricultural land shall be deemed to be polluted when the soil contains such quantities of harmful substances that its self-cleaning capacity is reduced, its physical, chemical and biotic properties deteriorate, plant growth and development is inhibited or prevented, groundwater or plants are polluted or the permanent soil fertility is reduced in any other manner due to harmful substances.
Permanent soil fertility is ensured when soil:
-         is not exposed to erosion;
-         is not compacted;
-         contains an adequate quantity of humus;
-         does not inhibit undisturbed plant growth;
-         enables the long-term development and quality of crops and forest plants (vegetation);
-         is capable of decomposing organic residues of plant and animal origin, animal or human excrement, residues of pesticides and other substances that return to the natural cycle of matter as secondary raw materials;
-         optimally absorbs, retains and releases water. The Government of the Republic of Slovenia (hereinafter: the Government) shall prescribe detailed criteria for permanent soil fertility and plant pollution and mandatory measures related thereto.
Article 4a
On land that is classified as agricultural land according to its designated and actual use, the planting of plantations of miscanthus and plantations of woody, shrub and tree species that are not intended for the cultivation of fruits and olives shall be permitted only when its land rating is less than 30.
A plantation under this Article shall mean a compact plantation of plants intended for commercial use with a density of 50 or more plants per hectare.
Article 5
Legal entities or natural persons that are the owners, lessees or other users of agricultural land (hereinafter: owner, lessee or other user of agricultural land) must allow other persons to carry out, in accordance with regulations, beekeeping, hunting and the recreational gathering of fruits of wild plants, herbaceous wild plants, mushrooms and wild animals on non-arable agricultural land owned or leased by them or otherwise allotted to them and allow other persons free access to non-arable agricultural land provided that no damage is caused by doing so.
Persons responsible for damage to land or crops shall be liable for compensation for damage to the owner, lessee or other user of agricultural land or crops in accordance with regulations.
Article 6
The provisions of this Act with regard to the use and cultivation of agricultural land shall also apply to land that is designated as construction land or non-agricultural land in a spatial planning document until a building permit or another relevant act is issued for such land in accordance with regulations.
Article 7
An owner, lessee or other user of agricultural land referred to in paragraph three of Article 2 of this Act must:
-         cultivate agricultural land with due diligence;
-         prevent the natural succession of vegetation on agricultural land, except on agricultural land that, in accordance with the regulation governing the records of actual land use, meets the conditions for land use classified as "trees and shrubs";
-         use farming methods appropriate for the land and location in order to prevent soil compaction, erosion and pollution and to ensure the permanent fertility of land.
For more efficient implementation of indent one of the preceding paragraph, notwithstanding the provisions of the Law of Property Code, the following shall be sufficient for the determination of agricultural land user and land use practices:
-         the consent of co-owners whose undivided interest represents more than half of the surface area of the land, if agricultural land is co-owned,
-         the consent of more than half of the owners if agricultural land is co-owned.
If the agricultural inspection authority establishes that an owner, lessee or other user of agricultural land fails to act in accordance with indent one of paragraph one of this Article, it shall impose the implementation of appropriate measures by a decision. An owner, lessee or other user of agricultural land must carry out such measures within the time limits set by the decision, which, however, may not exceed a period of one year.
If the agricultural inspection authority establishes that an owner, lessee or other user of agricultural land fails to act in accordance with indent two of paragraph one of this Article, it shall impose the implementation of appropriate measures by a decision. An owner, lessee or other user of agricultural land must carry out such measures no later than within a year of being served the decision; otherwise, the necessary measures shall be carried out at the expense of the owner, lessee or other user of agricultural land in an executive procedure.
If the implementation of the measures referred to in indent two of paragraph one of this Article cannot be imposed upon the owner of agricultural land because his or her residence is unknown and he or she has no legal representative, such land shall be transferred to the Fund for temporary management.
An agricultural inspector shall notify an administrative unit of the owner of agricultural land with an unknown residence and without a legal representative and carry out the procedure for the transfer of such land to the Fund for temporary management. An administrative unit shall submit a proposal to a social work centre that a trustee be appointed for this specific case, who shall participate in the procedure for the transfer of such land to the Fund for temporary management.
The right to the temporary management of agricultural land shall ex officio be entered in the land register on the basis of the final decision of an administrative unit.
An owner may submit an application to an administrative unit for the cessation of temporary management. Temporary management shall cease upon the decision on the cessation of temporary management becoming final. If such agricultural land has been leased out and the owner and lessee do not agree otherwise, the lease shall cease three years after the decision on the cessation of temporary management becomes final, while the lessee shall have the right to cultivate the land until the crops have been harvested, but no later than until the end of the calendar year.
In cases referred to in the preceding paragraph, the owner of agricultural land shall be obliged to refund to the Fund or possible lessee all expenses and investments related to the land; if the land was leased out, the owner shall be entitled to half of the rent over the lease period during temporary management.
The Government shall determine detailed criteria for assessing whether an owner, lessee or other user of agricultural land has acted with due diligence.
Persons who are not owners, lessees or other users of agricultural land shall use agricultural land on their own responsibility.



"""

# Split the document into chunks
chunks = document_text.split("\n\n")

# Transform the chunks into vectors using the embedding model
vectors = [model.encode(chunk) for chunk in chunks]

# Save the vectors to a vector database (for simplicity, we'll use a list)
vector_db = vectors

def user_query(query):
    # Find the best matches in the vector database
    query_vector = model.encode(query)
    similarities = [np.dot(query_vector, vector) for vector in vector_db]
    best_match_index = np.argmax(similarities)
    best_match_context = chunks[best_match_index]

    # Generate a prompt for GPT-3
    prompt = f"{best_match_context}\n{query}"

    # Get the answer from GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=60
    )

    return response.choices[0].text.strip()

# Example query
query = "What is Agricultural Land Act?"
print(user_query(query))
