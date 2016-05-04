from skimage.io import imshow, show

from utils.fio import get_image_roi
from utils.transcription import WordCoord

wordcoord = WordCoord('270-01-03')
roi = get_image_roi(wordcoord)
imshow(roi)

show()

print('done')
