# -*- coding: utf-8 -*-
from manim import *
import numpy as np
from inspect import signature

def animate(self, body, nazvy_bodu, baze):
    self.background_plane.add_coordinates()
    self.wait(2)
    x1 = self.add_vector(baze[0], color=BLUE)
    self.add_transformable_label(x1, MathTex(r"\vec{x}_1"), animate=False, at_tip=False, new_label=MathTex(r"\vec{x}_1"))
    x2 = self.add_vector(baze[1], color=YELLOW)
    self.add_transformable_label(x2, MathTex(r"\vec{x}_2"), animate=False, at_tip=False, new_label=MathTex(r"\vec{x}_2"))
    self.show_coordinates=True
    points = VGroup(*[Circle(radius=0.05, color=RED, fill_opacity=1).set_z_index(2).move_to((bod[0], bod[1], 0)) for bod in body])
    
    while len(nazvy_bodu) != len(body): 
            nazvy_bodu.append("")

    labels_array = [Tex(nazev).scale(0.7).move_to((*(np.array(bod) + 0.4*np.array(bod)/np.linalg.norm(bod)), 0)) for nazev, bod in zip(nazvy_bodu, body)]
    for label in labels_array: label.add_background_rectangle()
    
    basis_matrix = np.array([[baze[0][0], baze[1][0]], [baze[0][1], baze[1][1]]])
    transition_matrix = np.linalg.inv(basis_matrix)
    
    transformed_labels_positions = [np.matmul(transition_matrix,np.array(bod)) for bod in body]
    transformed_labels_array = [Tex(nazev).scale(0.7).move_to((*(pos + 0.4*pos/np.linalg.norm(pos)), 0)) for nazev, pos in zip(nazvy_bodu, transformed_labels_positions)]
    for label in transformed_labels_array: label.add_background_rectangle()
    labels = VGroup(*labels_array)
    transformed_labels =  VGroup(*transformed_labels_array)
    transform_labels = ReplacementTransform(labels, transformed_labels)
    
    lines = VGroup(*[Line(start=(*bod1, 0), end=(*bod2,0)).set_opacity(0.5) for bod1, bod2 in zip(body, body[1:] + [body[0]])])
    self.wait(1)
    self.play(FadeIn(points, labels, lines))
    self.add_transformable_mobject(points, lines)

    self.wait(2)
    self.apply_matrix(transition_matrix, added_anims=[transform_labels])
    self.wait(5)

def animate_curve(self, curve_func, baze):
    self.background_plane.add_coordinates()
    self.wait(2)
    x1 = self.add_vector(baze[0], color=BLUE)
    self.add_transformable_label(x1, MathTex(r"\vec{x}_1"), animate=False, at_tip=False, new_label=MathTex(r"\vec{x}_1"))
    x2 = self.add_vector(baze[1], color=YELLOW)
    self.add_transformable_label(x2, MathTex(r"\vec{x}_2"), animate=False, at_tip=False, new_label=MathTex(r"\vec{x}_2"))
    self.show_coordinates=True

    basis_matrix = np.array([[baze[0][0], baze[1][0]], [baze[0][1], baze[1][1]]])
    transition_matrix = np.linalg.inv(basis_matrix)
    if len(signature(curve_func).parameters) ==1:
        curve = ParametricFunction(curve_func, t_range = np.array([0, TAU]), fill_opacity=0).set_color(RED)
    elif len(signature(curve_func).parameters) ==2:
        curve = ImplicitFunction(curve_func).set_color(RED)
    else:
        raise Exception("Number of arguments not allowed")
    self.play(FadeIn(curve))
    self.add_transformable_mobject(curve)

    self.wait(2)
    self.apply_matrix(transition_matrix)
    self.wait(5)