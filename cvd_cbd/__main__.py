import sys
if len(sys.argv)>1 and sys.argv[1]=="study":
    from .study_cli import main
    raise SystemExit(main(sys.argv[2:]))
else:
    from .cli import main
    main()
