"""
=================
Random Forest GUI
=================

A simple graphical frontend for Random Forest mainly intended
for didactic purposes. You can create data points by point and click and
visualize the decision region induced by setting different number
of trees.

To create positive examples click the left mouse button; to create
negative examples click the right button.

If all examples are from the same class, it uses a one-class SVM.

"""
from __future__ import division

print __doc__

# Author: Peter Prettenhoer <peter.prettenhofer@gmail.com>
#         Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD Style.

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet

import Tkinter as Tk
import sys
import numpy as np

from sklearn import ensemble
from svm_gui import Model, get_parser

y_min, y_max = -50, 50
x_min, x_max = -50, 50


class Controller(object):
    def __init__(self, model):
        self.model = model
        self.n_estimators = Tk.IntVar()
        self.min_samples_leaf = Tk.IntVar()
        self.min_samples_split = Tk.IntVar()
        self.max_depth = Tk.IntVar()
        self.surface_type = Tk.IntVar()
        # Whether or not a model has been fitted
        self.fitted = False

    def fit(self):
        print "fit the model"
        train = np.array(self.model.data)
        X = train[:, 0:2]
        y = train[:, 2]

        n_estimators = int(self.n_estimators.get())
        clf = ensemble.RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)

        if hasattr(clf, 'score'):
            print "Accuracy:", clf.score(X, y) * 100
        X1, X2, Z = self.decision_surface(clf)
        self.model.clf = clf
        self.model.set_surface((X1, X2, Z))
        self.model.surface_type = self.surface_type.get()
        self.fitted = True
        self.model.changed("surface")

    def decision_surface(self, cls):
        delta = 1
        x = np.arange(x_min, x_max + delta, delta)
        y = np.arange(y_min, y_max + delta, delta)
        X1, X2 = np.meshgrid(x, y)
        # Z = cls.decision_function(np.c_[X1.ravel(), X2.ravel()])
        Z = cls.predict(np.c_[X1.ravel(), X2.ravel()])
        Z = Z.reshape(X1.shape)
        return X1, X2, Z

    def clear_data(self):
        self.model.data = []
        self.fitted = False
        self.model.changed("clear")

    def add_example(self, x, y, label):
        self.model.data.append((x, y, label))
        self.model.changed("example_added")

        # update decision surface if already fitted.
        self.refit()

    def refit(self):
        """Refit the model if already fitted. """
        if self.fitted:
            self.fit()


class View(object):
    """Test docstring. """
    def __init__(self, root, controller):
        f = Figure()
        ax = f.add_subplot(111)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        canvas.mpl_connect('button_press_event', self.onclick)
        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        self.controllbar = ControllBar(root, controller)
        self.f = f
        self.ax = ax
        self.canvas = canvas
        self.controller = controller
        self.contours = []
        self.c_labels = None

    def onclick(self, event):
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_example(event.xdata, event.ydata, 1)
            elif event.button == 3:
                self.controller.add_example(event.xdata, event.ydata, -1)

    def update_example(self, model, idx):
        x, y, l = model.data[idx]
        if l == 1:
            color = 'w'
        elif l == -1:
            color = 'k'
        self.ax.plot([x], [y], "%so" % color, scalex=0.0, scaley=0.0)

    def update(self, event, model):
        if event == "examples_loaded":
            for i in xrange(len(model.data)):
                self.update_example(model, i)

        if event == "example_added":
            self.update_example(model, -1)

        if event == "clear":
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.contours = []
            self.c_labels = None

        if event == "surface":
            self.remove_surface()
            self.plot_decision_surface(model.surface, model.surface_type)

        self.canvas.draw()

    def remove_surface(self):
        """Remove old decision surface."""
        if len(self.contours) > 0:
            for contour in self.contours:
                if isinstance(contour, ContourSet):
                    for lineset in contour.collections:
                        lineset.remove()
                else:
                    contour.remove()
            self.contours = []

    def plot_decision_surface(self, surface, type):
        X1, X2, Z = surface
        if type == 0:
            levels = [0.5]
            linestyles = ['solid']
            colors = 'k'
            self.contours.append(self.ax.contour(X1, X2, Z, levels,
                                                 colors=colors,
                                                 linestyles=linestyles))
        elif type == 1:
            self.contours.append(self.ax.contourf(X1, X2, Z, 10,
                                             cmap=matplotlib.cm.bone,
                                             origin='lower',
                                             alpha=0.85))
            self.contours.append(self.ax.contour(X1, X2, Z, [0.0],
                                                 colors='k',
                                                 linestyles=['solid']))
        else:
            raise ValueError("surface type unknown")


class ControllBar(object):
    def __init__(self, root, controller):
        fm = Tk.Frame(root)

        controller.n_estimators = Tk.StringVar()
        controller.n_estimators.set("1")
        Tk.Label(fm, text="number of trees:", width=14).pack(side=Tk.LEFT)
        Tk.Entry(fm, width=4,
                 textvariable=controller.n_estimators).pack(side=Tk.LEFT)

        controller.min_samples_leaf = Tk.StringVar()
        controller.min_samples_leaf.set("1")
        Tk.Label(fm, text="min leaf:", width=10).pack(side=Tk.LEFT)
        Tk.Entry(fm, width=4,
                 textvariable=controller.min_samples_leaf).pack(side=Tk.LEFT)

        controller.min_samples_split = Tk.StringVar()
        controller.min_samples_split.set("1")
        Tk.Label(fm, text="min split:", width=10).pack(side=Tk.LEFT)
        Tk.Entry(fm, width=4,
                 textvariable=controller.min_samples_split).pack(side=Tk.LEFT)

        controller.max_depth = Tk.StringVar()
        controller.max_depth.set("1")
        Tk.Label(fm, text="max depth:", width=10).pack(side=Tk.LEFT)
        Tk.Entry(fm, width=4,
                 textvariable=controller.max_depth).pack(side=Tk.LEFT)

        train_button = Tk.Button(fm, text='Fit', width=5,
                                 command=controller.fit)
        train_button.pack()
        fm.pack(side=Tk.LEFT)
        Tk.Button(fm, text='Clear', width=5,
                  command=controller.clear_data).pack(side=Tk.LEFT)


def main(argv):
    op = get_parser()
    opts, args = op.parse_args(argv[1:])
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("Scikit-learn Random Forest GUI")
    view = View(root, controller)
    model.add_observer(view)
    Tk.mainloop()

    if opts.output:
        model.dump_svmlight_file(opts.output)

if __name__ == "__main__":
    main(sys.argv)
