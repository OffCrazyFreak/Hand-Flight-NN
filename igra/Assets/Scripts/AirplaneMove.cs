using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using HeneGames.Airplane;
using UnityEngine.UIElements;
using System;
using System.IO;
using TMPro;

public class AirplaneMove : MonoBehaviour
{
    public float rotationRate = 2f;
    public float moveRate = 0.1f;
    public bool paused = false;
    public GameObject pauseUI;
    public TextMeshProUGUI debugText;
    private Rigidbody rb;
    private List<AirplaneCollider> airPlaneColliders = new List<AirplaneCollider>();
    private float angle = 0f;
    private float x = 0f;
    private float y = 0f;
    private bool planeIsDead = false;

    [Header("Colliders")]
    [SerializeField] private Transform crashCollidersRoot;
    [Header("Wing trail effects")]
    [Range(0.01f, 1f)]
    [SerializeField] private float trailThickness = 0.045f;
    [SerializeField] private TrailRenderer[] wingTrailEffects;
    [Header("Engine propellers settings")]
    [Range(10f, 10000f)]
    [SerializeField] private float propelSpeedMultiplier = 100f;

    [SerializeField] private GameObject[] propellers;

    [Header("Turbine light settings")]
    [Range(0.1f, 20f)]
    [SerializeField] private float turbineLightDefault = 1f;

    [Range(0.1f, 20f)]
    [SerializeField] private float turbineLightTurbo = 5f;

    [SerializeField] private Light[] turbineLights;

    // Start is called before the first frame update
    void Start()
    {
        //Get and set rigidbody
        rb = GetComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
        
        SetupColliders(crashCollidersRoot);

        Application.targetFrameRate = 30;
    }

    // Update is called once per frame
    void Update()
    {
        UpdatePropellersAndLights();

        string data;
        using (StreamReader sr = new StreamReader(Directory.GetCurrentDirectory() + "/shared.txt")) {
            data = sr.ReadToEnd();
        }
        debugText.text = "";
        string[] splitData = data.Split('\n');
        foreach (string line in splitData) {
            debugText.text += line + "\n";
        }
        if(int.Parse(splitData[0]) == 2) {
            paused = true;
            pauseUI.SetActive(true);
        }
        else {
            paused = false;
            pauseUI.SetActive(false);
            x = float.Parse(splitData[1]);
            y = float.Parse(splitData[2]);
            angle = float.Parse(splitData[3]);
        }

        if (paused) {
            return;
        }
        
        /*float inputH = Input.GetAxis("Horizontal");
        float inputV = Input.GetAxis("Vertical");

        if(Input.GetKey(KeyCode.Q)) {
            angle += rotationRate;
        }
        else if(Input.GetKey(KeyCode.E)) {
            angle -= rotationRate;
        }

        x += inputH * moveRate;
        y += inputV * moveRate;*/
        transform.rotation = Quaternion.Euler(Vector3.forward * angle);
        transform.localPosition = new Vector3(x, y, 0);

        if (!planeIsDead && HitSometing())
        {
            Crash();
        }
    }

    private void UpdatePropellersAndLights()
    {
        if(!planeIsDead)
        {
            //Rotate propellers if any
            if (propellers.Length > 0)
            {
                RotatePropellers(propellers, 10 * propelSpeedMultiplier);
            }

            //Control lights if any
            if (turbineLights.Length > 0)
            {
                ControlEngineLights(turbineLights, 0f);
            }
        }
        else
        {
            //Rotate propellers if any
            if (propellers.Length > 0)
            {
                RotatePropellers(propellers, 0f);
            }

            //Control lights if any
            if (turbineLights.Length > 0)
            {
                ControlEngineLights(turbineLights, 0f);
            }
        }
    }

    private void SetupColliders(Transform _root)
    {
        if (_root == null)
            return;

        //Get colliders from root transform
        Collider[] colliders = _root.GetComponentsInChildren<Collider>();

        //If there are colliders put components in them
        for (int i = 0; i < colliders.Length; i++)
        {
            //Change collider to trigger
            colliders[i].isTrigger = true;

            GameObject _currentObject = colliders[i].gameObject;

            //Add airplane collider to it and put it on the list
            AirplaneCollider _airplaneCollider = _currentObject.AddComponent<AirplaneCollider>();
            airPlaneColliders.Add(_airplaneCollider);

            //Add airplane conroller reference to collider
            _airplaneCollider.controller = this;

            //Add rigid body to it
            Rigidbody _rb = _currentObject.AddComponent<Rigidbody>();
            _rb.useGravity = false;
            _rb.isKinematic = true;
            _rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
        }
    }

    private void RotatePropellers(GameObject[] _rotateThese, float _speed)
    {
        for (int i = 0; i < _rotateThese.Length; i++)
        {
            _rotateThese[i].transform.Rotate(Vector3.forward * -_speed * Time.deltaTime);
        }
    }

    private void ControlEngineLights(Light[] _lights, float _intensity)
    {
        for (int i = 0; i < _lights.Length; i++)
        {
            if(!planeIsDead)
            {
                _lights[i].intensity = Mathf.Lerp(_lights[i].intensity, _intensity, 10f * Time.deltaTime);
            }
            else
            {
                _lights[i].intensity = Mathf.Lerp(_lights[i].intensity, 0f, 10f * Time.deltaTime);
            }
            
        }
    }

    private void ChangeWingTrailEffectThickness(float _thickness)
    {
        for (int i = 0; i < wingTrailEffects.Length; i++)
        {
            wingTrailEffects[i].startWidth = Mathf.Lerp(wingTrailEffects[i].startWidth, _thickness, Time.deltaTime * 10f);
        }
    }

    private bool HitSometing()
    {
        for (int i = 0; i < airPlaneColliders.Count; i++)
        {
            if (airPlaneColliders[i].collideSometing)
            {
                //Reset colliders
                foreach(AirplaneCollider _airPlaneCollider in airPlaneColliders)
                {
                    _airPlaneCollider.collideSometing = false;
                }

                return true;
            }
        }

        return false;
    }

    private void Crash()
    {
        planeIsDead = true;
    }
}
