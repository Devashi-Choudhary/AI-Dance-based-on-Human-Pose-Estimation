#The code is inspired from https://github.com/keshavoct98/DANCING-AI

'''Downloads videos from given links in "mp4" format and "480p" resolution.
Audio is extracted by converting downloaded videos to "wav" format.
All videos, audios are stored in "data" folder.'''

from pytube import YouTube
import moviepy.editor as mp

# Reading video links from "video_links.txt" file
with open('data/video_links.txt') as f:
    links = f.readlines()
    links = [x.strip() for x in links]


for i in range(0, len(links)):
    yt_obj = YouTube(links[i])
    
    print('\nAudio-'+str(i))
    audio = yt_obj.streams.filter().first()
    audio.download(filename=str(i), output_path='data/')

    video = mp.VideoFileClip('data/'+str(i)+'.mp4')
    video.audio.write_audiofile('data/'+str(i)+'.wav')  
    
    print('\nVideo-'+str(i))
    video = yt_obj.streams.filter(resolution='480p', mime_type="video/mp4").first()
    video.download(filename=str(i), output_path='data/')