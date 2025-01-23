using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AirplaneCollider : MonoBehaviour
{
    public bool collideSometing;

        [HideInInspector]
        public AirplaneMove controller;

        private void OnTriggerEnter(Collider other)
        {
            //Collide someting bad
            if(other.gameObject.GetComponent<AirplaneCollider>() == null)
            {
                collideSometing = true;
            }
        }
}
