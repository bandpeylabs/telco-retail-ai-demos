from pyvis.network import Network
import os
import uuid
import shutil

def displayGraph(graph):
    net = Network(
        height="750px", 
        width="100%", 
        directed=True, 
        cdn_resources='remote',
        notebook=True
    )

    net.options.groups = {
        "DIRECTIVE": {
            "icon": {
                "face": 'FontAwesome',
                "code": '\uf19c',
            }
        },
        "CHAPTER": {
            "icon": {
                "face": 'FontAwesome',
                "code": '\uf02d',
            }
        },
        "ARTICLE": {                 
            "icon": {
                "face": 'FontAwesome',
                "code": '\uf07c',
            }
        },
        "PARAGRAPH": {                 
            "icon": {
                "face": 'FontAwesome',
                "code": '\uf15b',
            }
        }
    }

    net.from_nx(graph)
    temp_html_path = f"/tmp/{uuid.uuid4().hex}.html"
    net.show(temp_html_path)
    
    # Move the file to a location where it can be downloaded
    download_path = f"/dbfs/FileStore/{uuid.uuid4().hex}.html"
    shutil.move(temp_html_path, download_path)
    
    return net.html.replace(
        '<head>',
        '<head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" type="text/css"/>'
    ), download_path