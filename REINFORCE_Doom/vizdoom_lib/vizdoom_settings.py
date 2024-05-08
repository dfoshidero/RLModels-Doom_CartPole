"""

This Python script is the settings the Reinforce algorithm, following RL Hugh's tutorial series on YouTube (2022). 
Ensure that the required libraries and VizDoom scenarios are correctly installed and configured to avoid runtime errors.

References:
Perkins, H., 2022. youtube-rl-demos/vizdoom at vizdoom18 · hughperkins/youtube-rl-demos [Online]. GitHub. Available from: https://github.com/hughperkins/youtube-rl-demos/tree/vizdoom18/vizdoom [Accessed 8 May 2024].
RL Hugh, n.d. ViZDoom: reinforcement learning using PyTorch - YouTube [Online]. www.youtube.com. Available from: https://www.youtube.com/playlist?list=PLdBvOJzNTtDUO4UC7R6N6_H-TFa78dka1 [Accessed 8 May 2024].

"""

import vizdoom as vzd

def setup_vizdoom(game: vzd.DoomGame):

    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)
    game.set_automap_buffer_enabled(True)
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(True)

    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  
    game.set_render_messages(False)  
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  