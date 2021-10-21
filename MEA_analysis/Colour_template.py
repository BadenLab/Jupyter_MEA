import numpy as np
import pandas as pd
import ipywidgets as widgets
import re


class Colour_template:

    nr_stimuli = 5
    stim_names = ["FFF_4_UV", "FFF_4", "FFF_6", "Silent_Substitution", "Contrast_Step"]
    indices = pd.Index(stim_names)
    silent_sub_bg = np.array([0, 2, 4, 6, 8, 9, 11, 13, 15, 17])
    silent_sub_cone = np.array([1, 3, 5, 7, 10, 12, 14, 16])
    plot_colour_dataframe = pd.DataFrame(
        columns=("Colours", "Description"), index=indices
    )

    # Colours
    Colours = []
    Colours.append(
        [
            "#fe7cfe",
            "#000000",
            "#7c86fe",
            "#000000",
            "#8afe7c",
            "#000000",
            "#fe7c7c",
            "#000000",
        ]
    )  # FFF_4_UV
    Colours.append(
        [
            "#7c86fe",
            "#000000",
            "#7cfcfe",
            "#000000",
            "#8afe7c",
            "#000000",
            "#fe7c7c",
            "#000000",
        ]
    )  # FFF_4
    # Colours.append(["#fe7cfe","#000000","#7c86fe","#000000","#7cfcfe","#000000","#8afe7c","#000000","#fafe7c","#000000","#fe7c7c","#000000"]) #FFF_6
    Colours.append(
        [
            "#fe7c7c",
            "#000000",
            "#fafe7c",
            "#000000",
            "#8afe7c",
            "#000000",
            "#7cfcfe",
            "#000000",
            "#7c86fe",
            "#000000",
            "#fe7cfe",
            "#000000",
        ]
    )  # FFF_6
    Colours.append(
        [
            "#E2E2E2",
            "#EF553B",
            "#E2E2E2",
            "#00CC96",
            "#E2E2E2",
            "#636EFA",
            "#E2E2E2",
            "#AB63FA",
            "#E2E2E2",
            "#7F7F7F",
            "#DC3912",
            "#7F7F7F",
            "#109618",
            "#7F7F7F",
            "#3366CC",
            "#7F7F7F",
            "#990099",
            "#7F7F7F",
        ]
    )  # Silent substitution
    Colours.append(
        [
            "#ffffff",
            "#000000",
            "#e6e6e6",
            "#000000",
            "#cdcdcd",
            "#000000",
            "#b4b4b4",
            "#000000",
            "#9b9b9b",
            "#000000",
            "#828282",
            "#000000",
            "#696969",
            "#000000",
            "#505050",
            "#000000",
            "#373737",
            "#000000",
            "#1e1e1e",
            "#000000",
        ]
    )  # Contrast Steps

    # Descriptions ("Names")
    Colours_names = []
    Colours_names.append(
        [
            "LED_630",
            "LED_630_OFF",
            "LED_505",
            "LED_505_OFF",
            "LED_420",
            "LED_420_OFF",
            "LED_360",
            "LED_360_OFF",
        ]
    )  # FFF_4_UV
    Colours_names.append(
        [
            "LED_630",
            "LED_630_OFF",
            "LED_505",
            "LED_505_OFF",
            "LED_480",
            "LED_480_OFF",
            "LED_420",
            "LED_420_OFF",
        ]
    )  # FFF_4
    Colours_names.append(
        [
            "LED_630",
            "LED_630_OFF",
            "LED_560",
            "LED_560_OFF",
            "LED_505",
            "LED_505_OFF",
            "LED_480",
            "LED_480_OFF",
            "LED_420",
            "LED_420_OFF",
            "LED_360",
            "LED_360_OFF",
        ]
    )  # FFF_6
    Colours_names.append(
        [
            "Background_On",
            "Red Cone OFF",
            "Background_Red_On",
            "Green Cone OFF",
            "Background_Green_On",
            "Blue Cone OFF",
            "Background_Blue_On",
            "VS Cone OFF",
            "Background_VS_On",
            "Background_Off",
            "Red Cone ON",
            "Background_Red_OFF",
            "Green Cone ON",
            "Background_Green_OFF",
            "Blue Cone ON",
            "Background_Blue_OFF",
            "VS Cone ON",
            "Background_VS_OFF",
        ]
    )  # Silent substitution
    Colours_names.append(
        [
            "100%_On",
            "100%_OFF",
            "90%_On",
            "90%_OFF",
            "80%_On",
            "80%_OFF",
            "70%_On",
            "70%_OFF",
            "60%_On",
            "60%_OFF",
            "50%_On",
            "50%_OFF",
            "40%_On",
            "40%_OFF",
            "30%_On",
            "30%_OFF",
            "20%_On",
            "20%_OFF",
            "10%_On",
            "10%_OFF",
        ]
    )  # Contrast Steps

    # Populate dataframe
    def __init__(self):
        for stimulus in range(self.nr_stimuli):
            self.plot_colour_dataframe.loc[self.stim_names[stimulus]] = [
                self.Colours[stimulus],
                self.Colours_names[stimulus],
            ]
        self.plot_colour_dataframe

        self.stimulus_select = widgets.RadioButtons(
            options=list(self.plot_colour_dataframe.index),
            layout={"width": "max-content"},
            description="Colourset",
            disabled=False,
        )

    def get_stimulus_colors(self, name, only_off=False, only_on=False):
        colours = np.asarray(self.plot_colour_dataframe.loc[name]["Colours"])
        if only_on:
            if name == "Silent_Substitution":
                colours = colours[self.silent_sub_cone]
            else:
                colours = colours[::2]
        if only_off:
            if name == "Silent_Substitution":
                colours = colours[self.silent_sub_bg]
            else:
                colours = colours[1::2]
        return colours

    def get_stimulus_names(self, name, only_off=False, only_on=False):
        names = np.asarray(self.plot_colour_dataframe.loc[name]["Description"])
        if only_on:
            if name == "Silent_Substitution":
                names = names[self.silent_sub_cone]
            else:
                names = names[::2]
        if only_off:
            if name == "Silent_Substitution":
                names = names[self.silent_sub_bg]
            else:
                names = names[1::2]
        return names

    def get_stimulus_wavelengths(self, name, only_off=False, only_on=False):
        names = self.get_stimulus_names(name, only_off=only_off, only_on=only_on)
        wavelengths = []
        for name in names:
            wavelengths.append(int(re.findall(r"\d+", name)[0]))
        return wavelengths

    def list_stimuli(self):
        return self.plot_colour_dataframe.index.to_numpy()

    # Widget stuff below:
    def select_preset_colour(self):
        return self.stimulus_select

    def pickstimcolour(self, selected_stimulus):
        self.selected_stimulus = selected_stimulus
        words = []
        nr_trigger = len(self.plot_colour_dataframe.loc[selected_stimulus]["Colours"])
        # print(nr_trigger)
        for trigger in range(nr_trigger):
            # print(trigger)
            selected_word = self.plot_colour_dataframe.loc[selected_stimulus][
                "Description"
            ][trigger]
            words.append(selected_word)

        items = [
            widgets.ColorPicker(
                description=w,
                value=self.plot_colour_dataframe.loc[selected_stimulus]["Colours"][
                    trigger
                ],
            )
            for w, trigger in zip(words, range(nr_trigger))
        ]
        first_box_items = []
        second_box_items = []
        third_box_items = []
        fourth_box_items = []

        a = 0
        for trigger in range(math.ceil(nr_trigger / 4)):
            try:
                first_box_items.append(items[a])
                a = a + 1
                second_box_items.append(items[a])
                a = a + 1
                third_box_items.append(items[a])
                a = a + 1
                fourth_box_items.append(items[a])
                a = a + 1
            except:
                break

        first_box = widgets.VBox(first_box_items)
        second_box = widgets.VBox(second_box_items)
        thrid_box = widgets.VBox(third_box_items)
        fourth_box = widgets.VBox(fourth_box_items)
        self.colour_select_box = widgets.HBox(
            [first_box, second_box, thrid_box, fourth_box]
        )
        return self.colour_select_box

    def changed_selection(self):
        self.axcolours = []
        self.LED_names = []
        nr_trigger = len(
            self.plot_colour_dataframe.loc[self.selected_stimulus]["Colours"]
        )
        for trigger in range(math.ceil(nr_trigger / 4)):
            try:
                self.axcolours.append(
                    self.colour_select_box.children[0].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[1].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[2].children[trigger].value
                )
                self.axcolours.append(
                    self.colour_select_box.children[3].children[trigger].value
                )

                self.LED_names = self.plot_colour_dataframe.loc[self.selected_stimulus][
                    "Description"
                ]
            except:

                break
