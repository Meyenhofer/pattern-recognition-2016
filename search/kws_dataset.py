from search.html import HTMLVisualization
from utils.transcription import WordCoord

def search_word(word, save=False, all_tooltips=True):
    """
    Finds all locations of a word. Optionally the locations can be saved in an
    HTML file with the same name as the word.
    """
    # TODO: Find the word. Currently just dummy data for testing.
    locations = ['270-01-01', '274-01-01', '270-02-02', '273-01-01', '273-02-02', '273-05-01', '270-09-01']
    if save:
        display_all_occurences(word, locations, output=word+'.html', all_tooltips=all_tooltips)

    return locations

def display_all_occurences(word, locations, output='default.html', all_tooltips=True):
    """
    Displays all images that are in the list of locations and highlights all
    locations within the images.
    """
    locations.sort()
    word_coords = [WordCoord(w) for w in locations]
    visual = HTMLVisualization(word)
    while word_coords:
        same_doc = 0
        curr_doc = word_coords[0].get_doc()
        for word in word_coords:
            if word.get_doc() == curr_doc:
                same_doc += 1
            else:
                break
        # remove the words of the same doc from the list
        word_ids = [str(w) for w in word_coords[:same_doc]]
        word_coords = word_coords[same_doc:]
        visual.add_image_by_id(curr_doc, word_ids=word_ids, all_tooltips=all_tooltips)
    visual.save(output)
