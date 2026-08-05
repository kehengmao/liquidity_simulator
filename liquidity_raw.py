from liba import quick_mt5_setup

if __name__ == "__main__":
    app = quick_mt5_setup()

    if app is not None:
        app.drawer.start(app.update)
