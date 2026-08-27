mention_start_token = "<Mention>"
mention_end_token = "</Mention>"
entity_end_token = "</Entity>"
title_end_token = "</Title>"
description_end_token = "</Description>"
type_sep_token = "<TypeSep>"
empty_description_token = "<EmptyDescription>"
empty_type_token = "<EmptyType>"

additional_tokens = [
                     mention_start_token,
                     mention_end_token,
                     entity_end_token,
                     title_end_token,
                     description_end_token,
                     type_sep_token,
                     empty_description_token,
                     empty_type_token,
                     ]

allowed_relations = [
                    "instance of", 
                    "occupation", 
                    "country", 
                    "country of citizenship", 
                    "country for sport", 
                    "country of origin", 
                    "sport",
                    "located in the administrative territorial entity", 
                    "continent", 
                    "genre", 
                    "subclass of", 
                    "part of", 
                    "home venue", 
                    "location"
                    ] 
