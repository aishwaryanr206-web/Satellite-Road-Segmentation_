from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from datetime import datetime

def create_pdf_report(output_path, image_name, stats, metrics=None):
    styles=getSampleStyleSheet()
    doc=SimpleDocTemplate(str(output_path),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=18*mm,bottomMargin=18*mm)
    story=[Paragraph("Satellite Road Segmentation Report",styles["Title"]),Spacer(1,8)]
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",styles["Normal"]))
    story.append(Paragraph(f"Input image: {image_name}",styles["Normal"])); story.append(Spacer(1,12))
    rows=[["Output statistic","Value"],["Road polygons",str(stats.get("polygon_count","N/A"))],
          ["Road area (pixels²)",f'{stats.get("road_area_pixels",0):.2f}'],
          ["Polygon perimeter (pixels)",f'{stats.get("polygon_perimeter_pixels",0):.2f}']]
    if metrics:
        rows += [["Pixel accuracy",metrics.get("pixel_accuracy","Not evaluated")],
                 ["Precision",metrics.get("precision","Not evaluated")],
                 ["F1 / Dice",metrics.get("dice","Not evaluated")]]
    if stats.get("georeferenced"):
        coordinate_note="GeoJSON and KML coordinates are transformed from the source raster CRS to longitude/latitude (EPSG:4326). Area and perimeter statistics remain in pixel units."
    else:
        coordinate_note="GIS exports use image pixel coordinates because the input has no georeferencing. Do not use these placeholder coordinates as real-world map locations."
    table=Table(rows,colWidths=[95*mm,65*mm])
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#19324d")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),.4,colors.grey),
        ("PADDING",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [table,Spacer(1,12),Paragraph("Method: ResNet-34 encoder + U-Net decoder; Focal-Tversky loss. "
        + coordinate_note,styles["BodyText"])]
    doc.build(story)
